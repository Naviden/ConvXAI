const messages = document.querySelector("#messages");
const options = document.querySelector("#options");
const composer = document.querySelector("#composer");
const question = document.querySelector("#question");
const reset = document.querySelector("#reset");
let sessionId = null;

function addMessage(role, text) {
  const element = document.createElement("div");
  element.className = `message ${role}`;
  element.textContent = text;
  messages.appendChild(element);
  messages.scrollTop = messages.scrollHeight;
}

function showOptions(values = []) {
  options.replaceChildren();
  values.forEach((value) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = value;
    button.addEventListener("click", () => send(value));
    options.appendChild(button);
  });
}

function setBusy(value) {
  question.disabled = value;
  composer.querySelector("button").disabled = value;
}

async function startSession() {
  setBusy(true);
  messages.replaceChildren();
  showOptions();
  try {
    const response = await fetch("/api/session", { method: "POST" });
    const payload = await response.json();
    sessionId = payload.session_id;
    addMessage("assistant", payload.response.message);
    showOptions(payload.response.options);
  } catch (error) {
    addMessage("assistant", `Could not start ConvXAI: ${error}`);
  } finally {
    setBusy(false);
    question.focus();
  }
}

async function send(text) {
  const clean = text.trim();
  if (!clean || !sessionId) return;
  addMessage("user", clean);
  showOptions();
  question.value = "";
  setBusy(true);
  try {
    const response = await fetch("/api/message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, text: clean }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Request failed");
    addMessage("assistant", payload.response.message);
    showOptions(payload.response.options);
  } catch (error) {
    addMessage("assistant", `Request failed: ${error.message}`);
  } finally {
    setBusy(false);
    question.focus();
  }
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  send(question.value);
});
reset.addEventListener("click", startSession);
startSession();
