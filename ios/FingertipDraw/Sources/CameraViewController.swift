import AVFoundation
import QuartzCore
import UIKit

final class CameraViewController: UIViewController {
    private let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "jp.ac.fingertipdepth.camera-session")
    private let captureQueue = DispatchQueue(label: "jp.ac.fingertipdepth.camera-frames")
    private lazy var previewLayer = AVCaptureVideoPreviewLayer(session: session)
    private let overlayView = OverlayView()
    private let statusLabel = UILabel()
    private let depthLabel = UILabel()
    private let drawButton = UIButton(type: .system)
    private let stopButton = UIButton(type: .system)
    private let clearButton = UIButton(type: .system)
    private let modelButton = UIButton(type: .system)
    private let trajectory = TrajectoryStore()
    private var selectedModel = StudentDepthModelVariant.defaultVariant
    private var pipeline: InferencePipeline?
    private var logger: SessionLogger?
    private var isDrawing = false
    private var isConfigured = false
    private var lastUpdateTime: CFTimeInterval?
    private var displayedFPS = 0.0

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        setupInterface()

        if let mismatch = TargetConfiguration.mismatchDescription() {
            showBlockingMessage(mismatch)
            return
        }
        do {
            pipeline = try InferencePipeline(initialModel: selectedModel)
            pipeline?.delegate = self
            logger = try SessionLogger(initialModel: selectedModel)
        } catch {
            showBlockingMessage(error.localizedDescription)
            return
        }
        requestCameraAndStart()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer.frame = view.bounds
        overlayView.frame = view.bounds
        view.bringSubviewToFront(modelButton)
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        sessionQueue.async { [session] in
            if session.isRunning { session.stopRunning() }
        }
    }

    private func setupInterface() {
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        overlayView.previewLayer = previewLayer
        view.addSubview(overlayView)

        statusLabel.numberOfLines = 2
        statusLabel.font = .monospacedSystemFont(ofSize: 12, weight: .medium)
        statusLabel.textColor = .white
        statusLabel.backgroundColor = UIColor.black.withAlphaComponent(0.62)
        statusLabel.layer.cornerRadius = 8
        statusLabel.layer.masksToBounds = true
        statusLabel.textAlignment = .center
        statusLabel.text = "カメラを準備中"

        depthLabel.font = .monospacedDigitSystemFont(ofSize: 25, weight: .bold)
        depthLabel.textColor = .white
        depthLabel.textAlignment = .center
        depthLabel.text = "— cm"

        configure(button: drawButton, title: "描画", color: .systemCyan, action: #selector(startDrawing))
        configure(button: stopButton, title: "停止", color: .systemOrange, action: #selector(stopDrawing))
        configure(button: clearButton, title: "消去", color: .systemGray, action: #selector(clearDrawing))
        let controls = UIStackView(arrangedSubviews: [drawButton, stopButton, clearButton])
        updateModelButton(for: selectedModel, isSwitching: false)
        modelButton.addTarget(self, action: #selector(toggleDepthModel), for: .touchUpInside)
        modelButton.accessibilityIdentifier = "depth-model-switch"
        controls.axis = .horizontal
        controls.spacing = 12
        controls.distribution = .fillEqually

        [statusLabel, depthLabel, modelButton, controls].forEach {
            $0.translatesAutoresizingMaskIntoConstraints = false
            view.addSubview($0)
        }
        NSLayoutConstraint.activate([
            statusLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 10),
            statusLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 12),
            statusLabel.widthAnchor.constraint(equalToConstant: 205),
            statusLabel.heightAnchor.constraint(greaterThanOrEqualToConstant: 46),
            depthLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            depthLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 14),
            modelButton.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 18),
            modelButton.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -18),
            modelButton.bottomAnchor.constraint(equalTo: controls.topAnchor, constant: -10),
            modelButton.heightAnchor.constraint(greaterThanOrEqualToConstant: 58),
            controls.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 18),
            controls.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -18),
            controls.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -14),
            controls.heightAnchor.constraint(equalToConstant: 52),
        ])
    }

    private func configure(
        button: UIButton,
        title: String,
        color: UIColor,
        action: Selector
    ) {
        var configuration = UIButton.Configuration.filled()
        configuration.title = title
        configuration.baseBackgroundColor = color.withAlphaComponent(0.88)
        configuration.baseForegroundColor = .white
        configuration.cornerStyle = .large
        button.configuration = configuration
        button.addTarget(self, action: action, for: .touchUpInside)
    }

    private func nextModel(after model: StudentDepthModelVariant) -> StudentDepthModelVariant? {
        let variants = StudentDepthModelVariant.allCases
        guard let currentIndex = variants.firstIndex(of: model) else { return nil }
        let nextIndex = variants.index(after: currentIndex)
        return nextIndex == variants.endIndex ? variants.first : variants[nextIndex]
    }

    private func updateModelButton(
        for model: StudentDepthModelVariant,
        isSwitching: Bool
    ) {
        let nextModel = nextModel(after: model)
        var configuration = UIButton.Configuration.filled()
        configuration.title = "推論モデル：\(model.descriptor.shortName)"
        configuration.subtitle = isSwitching
            ? "切り替え中…"
            : "タップで\(nextModel?.descriptor.shortName ?? "別モデル")へ切替"
        configuration.baseBackgroundColor = .systemIndigo
        configuration.baseForegroundColor = .white
        configuration.cornerStyle = .large
        modelButton.configuration = configuration
        modelButton.accessibilityLabel = "深さ推定モデル"
        modelButton.accessibilityValue = isSwitching
            ? "\(model.descriptor.displayName)へ切り替え中"
            : model.descriptor.displayName
        modelButton.accessibilityHint = isSwitching
            ? nil
            : nextModel.map {
                "ダブルタップで\($0.descriptor.displayName)へ切り替えます"
            }
    }

    private func requestCameraAndStart() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            configureAndStartCamera()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    granted ? self?.configureAndStartCamera()
                            : self?.showBlockingMessage("カメラの利用許可が必要です")
                }
            }
        default:
            showBlockingMessage("設定アプリでカメラの利用を許可してください")
        }
    }

    private func configureAndStartCamera() {
        guard !isConfigured else { return }
        isConfigured = true
        sessionQueue.async { [weak self] in
            guard let self else { return }
            do {
                try self.configureSession()
                self.session.startRunning()
                DispatchQueue.main.async {
                    self.statusLabel.text = "\(self.selectedModel.descriptor.shortName) · 検出待ち · 30 fps\n36 mm相当 / 固定カメラ"
                }
            } catch {
                DispatchQueue.main.async {
                    self.showBlockingMessage(error.localizedDescription)
                }
            }
        }
    }

    private func configureSession() throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = .hd1920x1080
        guard let device = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: .back
        ) else {
            throw CameraError.backWideCameraMissing
        }
        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else { throw CameraError.cannotAddInput }
        session.addInput(input)

        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }
        let requestedZoom = CGFloat(TargetConfiguration.requestedZoomFactor)
        guard requestedZoom >= device.minAvailableVideoZoomFactor,
              requestedZoom <= device.maxAvailableVideoZoomFactor else {
            throw CameraError.zoomUnavailable
        }
        device.videoZoomFactor = requestedZoom
        guard abs(device.videoZoomFactor - requestedZoom) <= 0.001 else {
            throw CameraError.zoomUnavailable
        }
        let duration = CMTime(value: 1, timescale: TargetConfiguration.captureFPS)
        let supports30FPS = device.activeFormat.videoSupportedFrameRateRanges.contains {
            $0.minFrameRate <= 30.0 && $0.maxFrameRate >= 30.0
        }
        guard supports30FPS else { throw CameraError.frameRateUnavailable }
        device.activeVideoMinFrameDuration = duration
        device.activeVideoMaxFrameDuration = duration
        device.automaticallyAdjustsVideoHDREnabled = false
        device.isVideoHDREnabled = false

        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ]
        output.setSampleBufferDelegate(self, queue: captureQueue)
        guard session.canAddOutput(output) else { throw CameraError.cannotAddOutput }
        session.addOutput(output)
        guard let connection = output.connection(with: .video) else {
            throw CameraError.videoConnectionMissing
        }
        guard connection.isVideoRotationAngleSupported(90) else {
            throw CameraError.portraitRotationUnavailable
        }
        connection.videoRotationAngle = 90
        if connection.isVideoStabilizationSupported {
            connection.preferredVideoStabilizationMode = .off
        }
        if connection.isCameraIntrinsicMatrixDeliverySupported {
            connection.isCameraIntrinsicMatrixDeliveryEnabled = true
        }

        DispatchQueue.main.async { [weak self] in
            guard let self, let previewConnection = self.previewLayer.connection else { return }
            if previewConnection.isVideoRotationAngleSupported(90) {
                previewConnection.videoRotationAngle = 90
            }
        }
    }

    @objc private func startDrawing() {
        isDrawing = true
        drawButton.configuration?.baseBackgroundColor = .systemGreen
        statusLabel.text = "描画中"
    }

    @objc private func stopDrawing() {
        isDrawing = false
        drawButton.configuration?.baseBackgroundColor = UIColor.systemCyan.withAlphaComponent(0.88)
        statusLabel.text = "停止中"
    }

    @objc private func clearDrawing() {
        trajectory.clear()
        if let current = overlayView.snapshot {
            overlayView.snapshot = OverlaySnapshot(
                selectedLandmarks: current.selectedLandmarks,
                currentPointM: current.currentPointM,
                trajectory: []
            )
        }
    }

    @objc private func toggleDepthModel() {
        guard let model = nextModel(after: selectedModel) else { return }
        guard model != selectedModel else { return }
        selectedModel = model
        modelButton.isEnabled = false
        updateModelButton(for: model, isSwitching: true)
        isDrawing = false
        drawButton.configuration?.baseBackgroundColor = UIColor.systemCyan.withAlphaComponent(0.88)
        trajectory.clear()
        overlayView.snapshot = nil
        depthLabel.text = "— cm"
        statusLabel.text = "\(model.descriptor.displayName)へ切替中"
        pipeline?.selectModel(model)
    }

    private func showBlockingMessage(_ message: String) {
        statusLabel.text = message
        statusLabel.backgroundColor = UIColor.systemRed.withAlphaComponent(0.80)
        drawButton.isEnabled = false
        stopButton.isEnabled = false
        modelButton.isEnabled = false
    }
}

extension CameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        pipeline?.enqueue(sampleBuffer)
    }
}

extension CameraViewController: InferencePipelineDelegate {
    func inferencePipeline(_ pipeline: InferencePipeline, didProduce update: InferenceUpdate) {
        guard update.modelVariant == selectedModel else {
            return
        }
        let now = CACurrentMediaTime()
        if let previous = lastUpdateTime {
            let instantaneous = 1.0 / max(now - previous, 1e-6)
            displayedFPS = displayedFPS == 0 ? instantaneous : displayedFPS * 0.85 + instantaneous * 0.15
        }
        lastUpdateTime = now
        let fingertip = update.landmarksXY[3]
        if isDrawing {
            trajectory.append(
                TrajectorySample(
                    timestampMS: update.timestampMS,
                    fingertipNormalized: fingertip,
                    cameraPointM: update.cameraPointM
                )
            )
        }
        overlayView.snapshot = OverlaySnapshot(
            selectedLandmarks: update.landmarksXY,
            currentPointM: update.cameraPointM,
            trajectory: trajectory.samples
        )
        depthLabel.text = String(format: "%.1f cm", update.cameraPointM.z * 100.0)
        statusLabel.backgroundColor = UIColor.black.withAlphaComponent(0.62)
        statusLabel.text = String(
            format: "%@ · %@ · %.1f fps\nhand %.1f / student %.1f / total %.1f ms",
            update.modelVariant.descriptor.shortName,
            update.intrinsics.source == .avFoundation ? "K:実測" : "K:近似",
            displayedFPS,
            update.handLatencyMS,
            update.studentLatencyMS,
            update.totalLatencyMS
        )
        logger?.append(update)
    }

    func inferencePipeline(_ pipeline: InferencePipeline, didActivate model: StudentDepthModelVariant) {
        guard model == selectedModel else { return }
        modelButton.isEnabled = true
        updateModelButton(for: model, isSwitching: false)
        depthLabel.text = "— cm"
        lastUpdateTime = nil
        displayedFPS = 0
        statusLabel.text = "\(model.descriptor.displayName)へ切替完了"
    }

    func inferencePipeline(_ pipeline: InferencePipeline, didRejectCaptureConfiguration message: String) {
        showBlockingMessage(message)
    }

    func inferencePipeline(_ pipeline: InferencePipeline, didFail message: String) {
        statusLabel.text = message
    }
}

enum CameraError: LocalizedError {
    case backWideCameraMissing
    case cannotAddInput
    case cannotAddOutput
    case videoConnectionMissing
    case zoomUnavailable
    case frameRateUnavailable
    case portraitRotationUnavailable

    var errorDescription: String? {
        switch self {
        case .backWideCameraMissing: return "背面広角カメラが見つかりません"
        case .cannotAddInput: return "カメラ入力を追加できません"
        case .cannotAddOutput: return "映像出力を追加できません"
        case .videoConnectionMissing: return "映像接続を作成できません"
        case .zoomUnavailable: return "36 mm相当のズームを設定できません"
        case .frameRateUnavailable: return "1080p 30 fpsを設定できません"
        case .portraitRotationUnavailable: return "縦向き映像を設定できません"
        }
    }
}
