const video = document.getElementById('videoElement');
const captureButton = document.getElementById('captureButton');
const capturedCanvas = document.getElementById('capturedCanvas');
const capturedImageInput = document.getElementById('capturedImage');
const captureForm = document.getElementById('captureForm');

let videoStream = null;

// Disable the capture button initially
captureButton.disabled = true;
captureButton.textContent = "Loading camera...";

// Request access to webcam
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function (stream) {
            videoStream = stream;
            video.srcObject = stream;

            // When video metadata is loaded, enable capture
            video.onloadedmetadata = () => {
                captureButton.disabled = false;
                captureButton.textContent = "Capture Photo";
            };
        })
        .catch(function (error) {
            console.error("Error accessing webcam:", error);
            captureButton.textContent = "Camera access denied";
        });
} else {
    captureButton.textContent = "Camera not supported";
}

captureButton.addEventListener('click', function () {
    // Set canvas size based on actual video dimensions
    capturedCanvas.width = video.videoWidth;
    capturedCanvas.height = video.videoHeight;

    const context = capturedCanvas.getContext('2d');
    context.drawImage(video, 0, 0, capturedCanvas.width, capturedCanvas.height);

    capturedCanvas.style.display = 'block';

    capturedCanvas.toBlob(function (blob) {
        const file = new File([blob], 'captured.jpg', { type: 'image/jpeg' });
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        capturedImageInput.files = dataTransfer.files;

        captureForm.style.display = 'block';
    }, 'image/jpeg');
});
