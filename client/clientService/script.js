const container = document.getElementById("container");
const registerbtn = document.getElementById("register");
const loginbtn = document.getElementById("login");

registerbtn.addEventListener("click", () => {
  container.classList.add("active");
});

loginbtn.addEventListener("click", () => {
  container.classList.remove("active");
});

// Get references to the input and button elements
const submitButton = document.getElementById('submitButton');
const userEmail = document.getElementById('emailId');
const responseMessage = document.getElementById('responseMessage');
const otpContainer = document.getElementById('otpcontainer');
// Event listener for button click
submitButton.addEventListener('click', async () => {
    // Get the value of the input field
    const inputData = userEmail.value;

    // Check if input is not empty
    if (inputData) {
        try {
            // API URL with dynamic email query parameter
            const apiURL = 'http://localhost:8080/user/verify-otp?email=' + inputData;

            // Make the POST request using fetch with async/await (no body, only query parameters)
            const response = await fetch(apiURL, {
                method: 'POST',
                headers: {
                    // No Content-Type needed since no body is being sent
                }
            });

            // Parse the JSON response
            const result = await response.json();
            // Display the response message
            if (result.status) {
                responseMessage.textContent = 'OTP Sent successfully! Response: ' + JSON.stringify(result);
                responseMessage.style.color = 'green';

                // Create the OTP input and submit button dynamically
                otpContainer.innerHTML = `
                    <input type="text" id="dynamicInput" placeholder="Enter OTP">
                    <button id="dynamicSubmit">Submit OTP</button>
                `;

                const submitButton = document.getElementById('dynamicSubmit');
                const inputField = document.getElementById('dynamicInput');

                // Handle OTP submission
                submitButton.addEventListener('click', async () => {
                    const inputOtp = inputField.value;
                    if (inputOtp) {
                        try {
                            // API URL with dynamic email and OTP query parameters
                            const apiURL = `http://localhost:8080/user/verify-otp?email=${inputData}&otp=${inputOtp}`;

                            // Make the POST request using fetch with async/await (no body, only query parameters)
                            const response = await fetch(apiURL, {
                                method: 'POST',
                                headers: {
                                    // No Content-Type needed since no body is being sent
                                }
                            });

                            // Parse the JSON response
                            const result = await response.json();
                            if (result.status) {
                                // Display the response
                                responseMessage.textContent = 'OTP Matched';
                                responseMessage.style.color = 'green';
                                window.location.href = 'fileUpload.html';
                            }
                        } catch (error) {
                            // Handle errors (e.g., network issues)
                            responseMessage.textContent = 'Error verifying OTP: ' + error.message;
                            responseMessage.style.color = 'red';
                        }
                    } else {
                        // If OTP input is empty, show an error message
                        responseMessage.textContent = 'Please enter the OTP!';
                        responseMessage.style.color = 'red';
                    }
                });
            }
        } catch (error) {
            // Handle errors (e.g., network issues)
            responseMessage.textContent = 'Error sending OTP: ' + error.message;
            responseMessage.style.color = 'red';
        }
    } else {
        // If email input is empty, show an error message
        responseMessage.textContent = 'Please enter an email!';
        responseMessage.style.color = 'red';
    }
});
