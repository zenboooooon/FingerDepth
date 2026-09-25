import Foundation

final class SessionLogger {
    private let queue = DispatchQueue(label: "jp.ac.fingertipdepth.logging")
    private let handle: FileHandle
    let directory: URL

    init() throws {
        let formatter = ISO8601DateFormatter()
        let name = formatter.string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        directory = documents.appendingPathComponent("FingertipDraw-\(name)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let metadata: [String: Any] = [
            "format": "fingertip-draw-ios-session",
            "format_version": 1,
            "target_device": TargetConfiguration.deviceName,
            "target_hardware_identifier": TargetConfiguration.hardwareIdentifier,
            "target_ios": TargetConfiguration.operatingSystemVersion,
            "actual_hardware_identifier": TargetConfiguration.currentHardwareIdentifier,
            "actual_ios": ProcessInfo.processInfo.operatingSystemVersionString,
            "checkpoint_sha256": TargetConfiguration.checkpointSHA256,
            "capture": [
                "width": TargetConfiguration.captureWidth,
                "height": TargetConfiguration.captureHeight,
                "fps": TargetConfiguration.captureFPS,
                "requested_zoom_factor": TargetConfiguration.requestedZoomFactor,
                "training_focal_length_px": TargetConfiguration.trainingFocalLengthPixels,
                "focal_length_relative_tolerance": TargetConfiguration.focalLengthRelativeTolerance,
                "stabilization": "off",
            ],
            "landmark_indices": [5, 6, 7, 8],
            "relative_z_used": false,
            "coordinate_convention": "x-right, y-down, z-forward",
        ]
        let metadataData = try JSONSerialization.data(
            withJSONObject: metadata,
            options: [.prettyPrinted, .sortedKeys]
        )
        try metadataData.write(to: directory.appendingPathComponent("session.json"), options: .atomic)

        let csvURL = directory.appendingPathComponent("trajectory.csv")
        FileManager.default.createFile(atPath: csvURL.path, contents: nil)
        handle = try FileHandle(forWritingTo: csvURL)
        let header = "timestamp_ms,u_px,v_px,x_m,y_m,z_m,fx_px,fy_px,cx_px,cy_px,intrinsics_source,handedness_score,hand_ms,student_ms,total_ms\n"
        try handle.write(contentsOf: Data(header.utf8))
    }

    deinit {
        queue.sync {}
        try? handle.close()
    }

    func append(_ update: InferenceUpdate) {
        let handedness = update.handednessScore.map { String(format: "%.6f", $0) } ?? ""
        let row = String(
            format: "%lld,%.1f,%.1f,%.8f,%.8f,%.8f,%.4f,%.4f,%.4f,%.4f,%@,%@,%.3f,%.3f,%.3f\n",
            Int64(update.timestampMS),
            update.fingertipPixel.x,
            update.fingertipPixel.y,
            update.cameraPointM.x,
            update.cameraPointM.y,
            update.cameraPointM.z,
            update.intrinsics.fx,
            update.intrinsics.fy,
            update.intrinsics.cx,
            update.intrinsics.cy,
            update.intrinsics.source.rawValue,
            handedness,
            update.handLatencyMS,
            update.studentLatencyMS,
            update.totalLatencyMS
        )
        queue.async { [handle = self.handle] in
            try? handle.write(contentsOf: Data(row.utf8))
        }
    }
}
