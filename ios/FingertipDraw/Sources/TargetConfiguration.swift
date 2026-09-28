import Foundation
import Darwin
import UIKit

enum TargetConfiguration {
    static let deviceName = "iPhone 15"
    static let hardwareIdentifier = "iPhone15,4"
    static let operatingSystemVersion = "26.6.1"
    static let captureWidth = 1080
    static let captureHeight = 1920
    static let captureFPS: Int32 = 30
    static let nativeEquivalentFocalLengthMM: Float = 26.0
    static let requestedEquivalentFocalLengthMM: Float = 36.0
    static let requestedZoomFactor = requestedEquivalentFocalLengthMM / nativeEquivalentFocalLengthMM
    static let trainingFocalLengthPixels: Float = 1832.9295592659223
    static let focalLengthRelativeTolerance: Float = 0.10

    static var currentHardwareIdentifier: String {
        var information = utsname()
        uname(&information)
        return withUnsafeBytes(of: &information.machine) { bytes in
            guard let baseAddress = bytes.baseAddress else { return "" }
            return String(cString: baseAddress.assumingMemoryBound(to: CChar.self))
        }
    }

    static func mismatchDescription() -> String? {
        #if targetEnvironment(simulator)
        return "実機のiPhone 15が必要です（現在はSimulatorです）"
        #else
        let actualDevice = currentHardwareIdentifier
        let actualOS = UIDevice.current.systemVersion
        guard actualDevice == hardwareIdentifier, actualOS == operatingSystemVersion else {
            return "対象: \(deviceName) / iOS \(operatingSystemVersion)\n現在: \(actualDevice) / iOS \(actualOS)"
        }
        return nil
        #endif
    }
}
