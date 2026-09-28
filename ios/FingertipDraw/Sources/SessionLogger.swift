import Foundation

final class SessionLogger {
    private let queue = DispatchQueue(label: "jp.ac.fingertipdepth.logging")
    private let handle: FileHandle
    let directory: URL

    init(initialModel: StudentDepthModelVariant = .defaultVariant) throws {
        let formatter = ISO8601DateFormatter()
        let name = formatter.string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        directory = documents.appendingPathComponent("FingertipDraw-\(name)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let availableModels: [[String: Any]] = StudentDepthModelVariant.allCases.map { variant in
            let descriptor = variant.descriptor
            var entry: [String: Any] = [
                "id": descriptor.id,
                "display_name": descriptor.displayName,
                "resource_name": descriptor.resourceName,
                "checkpoint_sha256": descriptor.checkpointSHA256,
            ]
            if let runManifestSHA256 = descriptor.runManifestSHA256 {
                entry["run_manifest_sha256"] = runManifestSHA256
            }
            if let exportBackend = descriptor.exportBackend {
                entry["export_backend"] = exportBackend
            }
            return entry
        }
        let metadata: [String: Any] = [
            "format": "fingertip-draw-ios-session",
            "format_version": 2,
            "target_device": TargetConfiguration.deviceName,
            "target_hardware_identifier": TargetConfiguration.hardwareIdentifier,
            "target_ios": TargetConfiguration.operatingSystemVersion,
            "actual_hardware_identifier": TargetConfiguration.currentHardwareIdentifier,
            "actual_ios": ProcessInfo.processInfo.operatingSystemVersionString,
            "initial_model_id": initialModel.descriptor.id,
            "available_models": availableModels,
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
        let header = "timestamp_ms,model_id,checkpoint_sha256,u_px,v_px,x_m,y_m,z_m,fx_px,fy_px,cx_px,cy_px,intrinsics_source,handedness_score,hand_ms,student_ms,total_ms\n"
        try handle.write(contentsOf: Data(header.utf8))
    }

    deinit {
        queue.sync {}
        try? handle.close()
    }

    func append(_ update: InferenceUpdate) {
        let handedness = update.handednessScore.map { String(format: "%.6f", $0) } ?? ""
        let row = String(
            format: "%lld,%@,%@,%.1f,%.1f,%.8f,%.8f,%.8f,%.4f,%.4f,%.4f,%.4f,%@,%@,%.3f,%.3f,%.3f\n",
            Int64(update.timestampMS),
            update.modelVariant.descriptor.id,
            update.modelVariant.descriptor.checkpointSHA256,
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
