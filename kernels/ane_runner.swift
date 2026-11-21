// Track E.4 — Native Swift CoreML / Apple Neural Engine (ANE) Runner & MLComputePlan Verifier
//
// Compile:
//   swiftc -O -framework CoreML -framework Foundation ane_runner.swift -o ane_runner
//
// Usage:
//   ./ane_runner --info
//   ./ane_runner --bench <model.mlmodelc> --units <ane|gpu|cpu|all> --c <C_in> --s <S> --iters <N> [--warmup <W>]
//   ./ane_runner --loop <model.mlmodelc> --units <ane|gpu|cpu|all> --c <C_in> --s <S> --seconds <T>

import CoreML
import Foundation

func deviceName(_ dev: MLComputeDevice) -> String {
    switch dev {
    case .cpu: return "CPU"
    case .gpu: return "GPU"
    case .neuralEngine(let ne): return "NeuralEngine(\(ne.totalCoreCount)cores)"
    @unknown default: return "Unknown"
    }
}

func parseUnits(_ s: String) -> MLComputeUnits {
    switch s.lowercased() {
    case "ane", "cpuandneuralengine": return .cpuAndNeuralEngine
    case "gpu", "cpuandgpu": return .cpuAndGPU
    case "cpu", "cpuonly": return .cpuOnly
    default: return .all
    }
}

func makeInput(c: Int, s: Int, useFP16: Bool) throws -> MLFeatureProvider {
    let shape: [NSNumber] = [NSNumber(value: c), 1, NSNumber(value: s)]
    let dtype: MLMultiArrayDataType = useFP16 ? .float16 : .float32
    let arr = try MLMultiArray(shape: shape, dataType: dtype)
    let count = c * s
    if useFP16 {
        let ptr = arr.dataPointer.assumingMemoryBound(to: UInt16.self)
        // 0x2e66 ~ 0.1 in IEEE-754 float16
        for i in 0..<count { ptr[i] = 0x2e66 }
    } else {
        let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<count { ptr[i] = 0.1 }
    }
    return try MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: arr)])
}

func inspectComputePlan(url: URL, config: MLModelConfiguration) -> [String: Any] {
    var result: [String: Any] = ["status": "unavailable"]
    let sem = DispatchSemaphore(value: 0)
    Task {
        do {
            let plan = try await MLComputePlan.load(contentsOf: url, configuration: config)
            var layers: [[String: Any]] = []
            var aneCount = 0
            var gpuCount = 0
            var cpuCount = 0
            switch plan.modelStructure {
            case .neuralNetwork(let nn):
                for layer in nn.layers {
                    let usage = plan.deviceUsage(for: layer)
                    let pref = usage.map { deviceName($0.preferred) } ?? "unknown"
                    let supp = usage.map { $0.supported.map { deviceName($0) } } ?? []
                    if pref.hasPrefix("NeuralEngine") { aneCount += 1 }
                    else if pref == "GPU" { gpuCount += 1 }
                    else if pref == "CPU" { cpuCount += 1 }
                    layers.append([
                        "name": layer.name,
                        "type": layer.type,
                        "preferred": pref,
                        "supported": supp,
                    ])
                }
            case .program(let prog):
                if let mainFn = prog.functions["main"] {
                    for op in mainFn.block.operations {
                        let usage = plan.deviceUsage(for: op)
                        let pref = usage.map { deviceName($0.preferred) } ?? "unknown"
                        let supp = usage.map { $0.supported.map { deviceName($0) } } ?? []
                        if pref.hasPrefix("NeuralEngine") { aneCount += 1 }
                        else if pref == "GPU" { gpuCount += 1 }
                        else if pref == "CPU" { cpuCount += 1 }
                        layers.append([
                            "name": op.operatorName,
                            "type": "mlprogram",
                            "preferred": pref,
                            "supported": supp,
                        ])
                    }
                }
            default:
                break
            }
            result = [
                "status": "ok",
                "ane_layers": aneCount,
                "gpu_layers": gpuCount,
                "cpu_layers": cpuCount,
                "layers": layers,
            ]
        } catch {
            result = ["status": "error", "error": "\(error)"]
        }
        sem.signal()
    }
    sem.wait()
    return result
}

func emitJSON(_ obj: Any) {
    if let data = try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys]),
       let str = String(data: data, encoding: .utf8) {
        print(str)
    }
}

// Parse CLI args
let argv = CommandLine.arguments
func getArg(_ flag: String, _ def: String = "") -> String {
    if let idx = argv.firstIndex(of: flag), idx + 1 < argv.count {
        return argv[idx + 1]
    }
    return def
}
func hasFlag(_ flag: String) -> Bool {
    return argv.contains(flag)
}

if hasFlag("--info") {
    let devices = MLModel.availableComputeDevices.map { deviceName($0) }
    var neCores = 0
    for d in MLModel.availableComputeDevices {
        if case .neuralEngine(let ne) = d { neCores = ne.totalCoreCount }
    }
    emitJSON([
        "available_compute_devices": devices,
        "neural_engine_cores": neCores,
    ])
    exit(0)
}

let modelPath = hasFlag("--bench") ? getArg("--bench") : getArg("--loop")
let unitsStr = getArg("--units", "ane")
let cIn = Int(getArg("--c", "4096")) ?? 4096
let sLen = Int(getArg("--s", "1")) ?? 1
let iters = Int(getArg("--iters", "30")) ?? 30
let warmup = Int(getArg("--warmup", "5")) ?? 5
let loopSec = Double(getArg("--seconds", "0")) ?? 0.0
let useFP16 = hasFlag("--fp16")

let url = URL(fileURLWithPath: modelPath)
let config = MLModelConfiguration()
config.computeUnits = parseUnits(unitsStr)

do {
    let tLoad0 = CFAbsoluteTimeGetCurrent()
    let model = try MLModel(contentsOf: url, configuration: config)
    let loadMs = (CFAbsoluteTimeGetCurrent() - tLoad0) * 1000.0
    let input = try makeInput(c: cIn, s: sLen, useFP16: useFP16)

    // Warmup + verify non-zero output
    var sampleVal: Double = 0.0
    for _ in 0..<warmup {
        let out = try model.prediction(from: input)
        if let fv = out.featureValue(for: "y"), let arr = fv.multiArrayValue, arr.count > 0 {
            let v = arr[0].doubleValue
            sampleVal = v.isFinite ? v : -999.0
        }
    }

    if loopSec > 0.0 {
        let t0 = CFAbsoluteTimeGetCurrent()
        var n = 0
        while CFAbsoluteTimeGetCurrent() - t0 < loopSec {
            _ = try model.prediction(from: input)
            n += 1
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        emitJSON([
            "model": modelPath,
            "units": unitsStr,
            "iters": n,
            "elapsed_s": elapsed,
            "mean_ms": (elapsed * 1000.0) / Double(max(n, 1)),
            "sample_out0": sampleVal,
        ])
        exit(0)
    }

    let planInfo = inspectComputePlan(url: url, config: config)
    var timesMs: [Double] = []
    timesMs.reserveCapacity(iters)
    for _ in 0..<iters {
        let t0 = CFAbsoluteTimeGetCurrent()
        _ = try model.prediction(from: input)
        let dt = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
        timesMs.append(dt)
    }
    timesMs.sort()
    let medMs = timesMs[timesMs.count / 2]
    let minMs = timesMs.first ?? 0.0
    let p90Ms = timesMs[Int(Double(timesMs.count - 1) * 0.9)]

    emitJSON([
        "model": modelPath,
        "units": unitsStr,
        "c_in": cIn,
        "s_len": sLen,
        "load_ms": loadMs,
        "iters": iters,
        "median_ms": medMs,
        "min_ms": minMs,
        "p90_ms": p90Ms,
        "sample_out0": sampleVal,
        "compute_plan": planInfo,
    ])
} catch {
    emitJSON(["error": "\(error)", "model": modelPath, "units": unitsStr])
    exit(1)
}
