document.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById("chatModal");
    const btn = document.getElementById("openChatBtn");
    const span = document.getElementsByClassName("close")[0];
    const sendBtn = document.getElementById("sendBtn");
    const userInput = document.getElementById("userInput");
    const chatBody = document.getElementById("chatBody");

    btn.onclick = function() {
        modal.style.display = modal.style.display === "block" ? "none" : "block";
    }

    span.onclick = function() {
        modal.style.display = "none";
    }

    window.onclick = function(event) {
        if (event.target == modal) {
            modal.style.display = "none";
        }
    }

    sendBtn.onclick = function() {
        sendMessage();
    }

    userInput.addEventListener("keypress", function(event) {
        if (event.key === "Enter") {
            event.preventDefault();
            sendMessage();
        }
    });

    function sendMessage() {
        const userMessage = userInput.value.trim();
        if (userMessage !== "") {
            const userMessageDiv = document.createElement("div");
            userMessageDiv.className = "message user-message";
            userMessageDiv.textContent = userMessage;
            chatBody.appendChild(userMessageDiv);
            userInput.value = "";
            chatBody.scrollTop = chatBody.scrollHeight;

            fetch('/chatbot/ai-response/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCookie('csrftoken')
                },
                body: JSON.stringify({ message: userMessage })
            })
            .then(response => response.json())
            .then(data => {
                const botMessageDiv = document.createElement("div");
                botMessageDiv.className = "message bot-message";
                botMessageDiv.textContent = data.message;
                chatBody.appendChild(botMessageDiv);
                chatBody.scrollTop = chatBody.scrollHeight;
            })
            .catch(error => {
                console.error('Error:', error);
            });
        }
    }

    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
});
