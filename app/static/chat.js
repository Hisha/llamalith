const chatBox = document.getElementById('chat-box');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const modelSelect = document.getElementById('model-select');
const systemPrompt = document.getElementById('system-prompt');
const sessionId = localStorage.getItem('session_id') || crypto.randomUUID();
localStorage.setItem('session_id', sessionId);

function appendMessage(role, content) {
  const div = document.createElement('div');
  div.className = `mb-2 ${role === 'user' ? 'text-green-400' : 'text-yellow-300'}`;
  div.textContent = `${role}: ${content}`;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

chatForm.onsubmit = async (e) => {
  e.preventDefault();
  const input = userInput.value.trim();
  if (!input) return;
  appendMessage('user', input);
  userInput.value = '';

  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_input: input,
      model: modelSelect.value,
      system_prompt: systemPrompt.value,
      session_id: sessionId
    })
  });

  const data = await res.json();
  appendMessage('assistant', data.reply);
};
