import Foundation

struct TrajectorySample {
    let timestampMS: Int
    let fingertipNormalized: SIMD2<Float>
    let cameraPointM: SIMD3<Float>
}

final class TrajectoryStore {
    private(set) var samples: [TrajectorySample] = []
    var maximumCount = 3_600

    func append(_ sample: TrajectorySample) {
        samples.append(sample)
        if samples.count > maximumCount {
            samples.removeFirst(samples.count - maximumCount)
        }
    }

    func clear() {
        samples.removeAll(keepingCapacity: true)
    }
}
