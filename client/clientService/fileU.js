const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewImage = document.getElementById('previewImage');
const imageName = document.getElementById('imageName');
const imageSize = document.getElementById('imageSize');
const imageInfo = document.getElementById('imageInfo');
const submitButton = document.getElementById('submitButton');
const message = document.getElementById('message');

// File input change event
fileInput.addEventListener('change', function() {
    const file = this.files[0];
    handleFile(file);
});

// Drag and Drop functionality
uploadArea.addEventListener('dragover', (event) => {
    event.preventDefault();
    uploadArea.classList.add('active');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('active');
});

uploadArea.addEventListener('drop', (event) => {
    event.preventDefault();
    uploadArea.classList.remove('active');
    const file = event.dataTransfer.files[0];
    handleFile(file);
});

// Handle file and display preview
function handleFile(file) {
    if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            previewImage.classList.remove('hidden');
            imageInfo.classList.remove('hidden');
            submitButton.classList.remove('hidden');
            imageName.textContent = file.name;
            imageSize.textContent = (file.size / 1024).toFixed(2) + ' KB';
        };
        reader.readAsDataURL(file);
    } else {
        alert('Please upload a valid image file.');
    }
}

// Submit button click event
submitButton.addEventListener('click', function() {
    // Simulating form submission process
    submitButton.textContent = 'Uploading...';
    submitButton.disabled = true;
    
    // Simulating a 2-second delay to mimic an upload process
    setTimeout(() => {
        submitButton.textContent = 'Submit';
        submitButton.disabled = false;
        message.classList.remove('hidden');
        message.textContent = 'Image uploaded successfully!';
        message.style.color = 'green';
    }, 2000);
});