/**
 * RAG System — Professional Client Application Logic
 * Supports streaming chat, document management, filter searching, settings sync, and rich markdown formatting.
 */

// ============================================================
// State Management
// ============================================================
let isStreaming = false;
let allDocuments = [];
const API_BASE_URL = (window.RAG_API_BASE_URL || "").replace(/\/$/, "");

function apiUrl(path) {
    return `${API_BASE_URL}${path}`;
}

// ============================================================
// Initialization
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
    setupChatInput();
    setupFileUpload();
    loadDocuments();
    loadSettings();
});

// ============================================================
// Chat Handler
// ============================================================
function setupChatInput() {
    const input = document.getElementById("chatInput");
    if (!input) return;

    // Auto-resize textarea dynamically
    input.addEventListener("input", () => {
        input.style.height = "auto";
        input.style.height = Math.min(input.scrollHeight, 140) + "px";
        updateCharCount();
    });

    // Enter to send, Shift+Enter for new line
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
}

function updateCharCount() {
    const input = document.getElementById("chatInput");
    const counter = document.getElementById("charCount");
    if (!input || !counter) return;
    const len = input.value.length;
    counter.textContent = len > 0 ? `${len} chars` : "";
}

function sendQuickPrompt(promptText) {
    const input = document.getElementById("chatInput");
    if (!input || isStreaming) return;
    input.value = promptText;
    sendMessage();
}

function clearChat() {
    const container = document.getElementById("chatMessages");
    if (!container) return;

    container.innerHTML = `
        <div class="welcome-screen" id="welcomeScreen">
            <div class="welcome-badge">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
                <span>LangChain + AI Vector Pipeline</span>
            </div>
            
            <h1 class="welcome-title">AI Document Intelligence</h1>
            <p class="welcome-subtitle">
                Upload complex business documents, research papers, or specs. Ask detailed questions or click a starter prompt below for an instant summary.
            </p>

            <div class="starter-prompts">
                <div class="prompt-card" onclick="sendQuickPrompt('Summarize the uploaded document concisely highlighting key objectives, findings, and conclusion.')">
                    <div class="prompt-icon">📄</div>
                    <div class="prompt-text">
                        <div class="prompt-title">Summarize Document <span class="kbd-hint">⏎</span></div>
                        <div class="prompt-desc">Objectives, key findings & conclusion</div>
                    </div>
                </div>
                <div class="prompt-card" onclick="sendQuickPrompt('What are the main topics and key takeaways discussed in this document?')">
                    <div class="prompt-icon">💡</div>
                    <div class="prompt-text">
                        <div class="prompt-title">Key Takeaways <span class="kbd-hint">⏎</span></div>
                        <div class="prompt-desc">Core concepts and major highlights</div>
                    </div>
                </div>
                <div class="prompt-card" onclick="sendQuickPrompt('Extract critical statistics, data tables, metrics, and quantitative findings.')">
                    <div class="prompt-icon">📊</div>
                    <div class="prompt-text">
                        <div class="prompt-title">Extract Metrics & Data <span class="kbd-hint">⏎</span></div>
                        <div class="prompt-desc">Statistics, measurements & tables</div>
                    </div>
                </div>
                <div class="prompt-card" onclick="sendQuickPrompt('What methodologies, recommendations, or decisions are detailed in the document?')">
                    <div class="prompt-icon">🛠️</div>
                    <div class="prompt-text">
                        <div class="prompt-title">Methodologies & Decisions <span class="kbd-hint">⏎</span></div>
                        <div class="prompt-desc">Execution methods & guidance</div>
                    </div>
                </div>
            </div>
        </div>
    `;

    showToast("Conversation reset", "info");
}

async function sendMessage() {
    const input = document.getElementById("chatInput");
    const question = input.value.trim();
    if (!question || isStreaming) return;

    // Hide welcome screen if present
    const welcome = document.getElementById("welcomeScreen");
    if (welcome) welcome.remove();

    // Add user message bubble
    appendMessage("user", question);
    input.value = "";
    input.style.height = "auto";

    // Add assistant placeholder with glowing typing dots
    const assistantEl = appendMessage("assistant", "");
    const contentEl = assistantEl.querySelector(".message-content");
    contentEl.innerHTML = `<div class="typing-indicator"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>`;

    isStreaming = true;
    updateSendButton();
    updateSystemStatus("Generating...");

    try {
        const response = await fetch(apiUrl("/api/chat"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Server error");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = "";
        let sourcesHtml = "";
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop(); // Keep incomplete line in buffer

            for (const line of lines) {
                if (!line.startsWith("data: ")) continue;

                try {
                    const event = JSON.parse(line.slice(6));

                    if (event.type === "token") {
                        fullText += event.content;
                        contentEl.innerHTML = renderMarkdown(fullText);
                        scrollToBottom();
                    } else if (event.type === "sources") {
                        sourcesHtml = renderSources(event.content);
                    } else if (event.type === "done") {
                        if (sourcesHtml) {
                            contentEl.innerHTML = renderMarkdown(fullText) + sourcesHtml;
                        }
                    } else if (event.type === "error") {
                        contentEl.innerHTML = `<div class="error-message" style="color: #f87171; background: rgba(239, 68, 68, 0.1); padding: 12px; border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.3);">⚠️ ${escapeHtml(event.content)}</div>`;
                    }
                } catch {
                    // Skip malformed JSON lines
                }
            }
        }

        if (fullText) {
            contentEl.innerHTML = renderMarkdown(fullText) + sourcesHtml;
        }

    } catch (err) {
        contentEl.innerHTML = `<div class="error-message" style="color: #f87171; background: rgba(239, 68, 68, 0.1); padding: 12px; border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.3);">⚠️ ${escapeHtml(err.message)}</div>`;
    }

    isStreaming = false;
    updateSendButton();
    updateSystemStatus("System Ready");
    scrollToBottom();
}

function appendMessage(role, content) {
    const container = document.getElementById("chatMessages");
    const avatar = role === "user" ? 
        `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>` : 
        `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>`;

    const messageEl = document.createElement("div");
    messageEl.className = `message ${role}`;
    messageEl.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content-box">
            <div class="message-content">${role === "user" ? escapeHtml(content) : content}</div>
        </div>
    `;

    container.appendChild(messageEl);
    scrollToBottom();
    return messageEl;
}

function updateSendButton() {
    const btn = document.getElementById("sendBtn");
    if (!btn) return;
    btn.disabled = isStreaming;
}

function updateSystemStatus(text) {
    const statsText = document.getElementById("statsText");
    if (statsText) statsText.textContent = text;
}

function scrollToBottom() {
    const container = document.getElementById("chatMessages");
    if (container) container.scrollTop = container.scrollHeight;
}

// ============================================================
// Markdown Renderer with Tables & Copy Code Support
// ============================================================
function renderMarkdown(text) {
    let html = escapeHtml(text);

    // Fenced Code Blocks (```lang ... ```)
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
        const langName = lang || "code";
        const codeId = "code_" + Math.random().toString(36).substr(2, 9);
        return `<div class="code-block-wrapper">
            <div class="code-header">
                <span>${langName}</span>
                <button class="btn-copy-code" onclick="copyCode('${codeId}')">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                    Copy
                </button>
            </div>
            <pre><code id="${codeId}">${code.trim()}</code></pre>
        </div>`;
    });

    // Inline Code
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

    // Italic
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

    // Headers
    html = html.replace(/^#### (.+)$/gm, "<h4>$1</h4>");
    html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
    html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");

    // Tables
    html = html.replace(/^\|(.+)\|$/gm, (match) => {
        const cells = match.split("|").slice(1, -1).map(c => c.trim());
        const isHeader = cells.every(c => /^:?-+:?$/.test(c));
        if (isHeader) return "<!-- TABLE_SEP -->";
        const cellTag = "td";
        return `<tr>${cells.map(c => `<${cellTag}>${c}</${cellTag}>`).join("")}</tr>`;
    });
    html = html.replace(/(?:<tr>.+<\/tr>\n?)+/g, (tableContent) => {
        const rows = tableContent.split("\n").filter(Boolean);
        if (rows.length > 0) {
            // First row as header
            const firstRow = rows[0].replace(/<td>/g, "<th>").replace(/<\/td>/g, "</th>");
            const bodyRows = rows.slice(2).join(""); // Skip header and divider row
            return `<table><thead>${firstRow}</thead><tbody>${bodyRows}</tbody></table>`;
        }
        return tableContent;
    });

    // Unordered Lists
    html = html.replace(/^[\-\*] (.+)$/gm, "<li>$1</li>");
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, "<ul>$1</ul>");

    // Ordered Lists
    html = html.replace(/^\d+\. (.+)$/gm, "<li>$1</li>");

    // Paragraphs
    html = html.replace(/\n\n/g, "</p><p>");
    html = html.replace(/\n/g, "<br>");
    html = `<p>${html}</p>`;

    // Clean up empty paragraphs wrapped around block elements
    html = html.replace(/<p><\/p>/g, "");
    html = html.replace(/<p>(<h[1-4]>)/g, "$1");
    html = html.replace(/(<\/h[1-4]>)<\/p>/g, "$1");
    html = html.replace(/<p>(<div class="code-block-wrapper">)/g, "$1");
    html = html.replace(/(<\/div>)<\/p>/g, "$1");
    html = html.replace(/<p>(<table>)/g, "$1");
    html = html.replace(/(<\/table>)<\/p>/g, "$1");
    html = html.replace(/<p>(<ul>)/g, "$1");
    html = html.replace(/(<\/ul>)<\/p>/g, "$1");

    return html;
}

function copyCode(elementId) {
    const codeEl = document.getElementById(elementId);
    if (!codeEl) return;
    navigator.clipboard.writeText(codeEl.textContent).then(() => {
        showToast("Code copied to clipboard!", "success");
    });
}

// ============================================================
// Source Citation Renderer & Popover
// ============================================================
function renderSources(sources) {
    if (!sources || sources.length === 0) return "";

    const chips = sources
        .map((s) => {
            const name = escapeHtml(s.document_name);
            const score = s.similarity_score ? `${(s.similarity_score * 100).toFixed(0)}%` : "";
            const text = escapeHtml(s.chunk_text || "");
            const page = s.page_number ? ` · p.${s.page_number}` : "";
            return `<span class="source-chip"
                        onmouseenter="showSourcePopover(event, '${name}${page}', \`${text.replace(/`/g, "\\`").replace(/\\/g, "\\\\")}\`)"
                        onmouseleave="hideSourcePopover()">
                        📄 ${name}${page}
                        <span class="score">${score}</span>
                    </span>`;
        })
        .join("");

    return `<div class="sources-container">
                <div class="sources-label">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
                    Source Citations
                </div>
                <div>${chips}</div>
            </div>`;
}

function showSourcePopover(event, title, content) {
    const popover = document.getElementById("sourcePopover");
    if (!popover) return;
    document.getElementById("popoverTitle").textContent = title;
    document.getElementById("popoverContent").textContent = content;

    const rect = event.target.getBoundingClientRect();
    popover.style.left = Math.min(rect.left, window.innerWidth - 440) + "px";
    popover.style.top = (rect.top - 12) + "px";
    popover.style.transform = "translateY(-100%)";
    popover.classList.add("visible");
}

function hideSourcePopover() {
    const popover = document.getElementById("sourcePopover");
    if (popover) popover.classList.remove("visible");
}

// ============================================================
// File Upload Integration
// ============================================================
function setupFileUpload() {
    const dropzone = document.getElementById("uploadDropzone");
    const fileInput = document.getElementById("fileInput");
    if (!dropzone || !fileInput) return;

    const chatAttachButton = document.getElementById("chatAttachBtn");
    chatAttachButton?.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("drag-over");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("drag-over");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("drag-over");
        if (e.dataTransfer.files.length > 0) uploadFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            uploadFiles(fileInput.files);
            fileInput.value = "";
        }
    });
}

async function uploadFiles(files) {
    const progressEl = document.getElementById("uploadProgress");
    const barFill = document.getElementById("progressBarFill");
    const statusEl = document.getElementById("uploadStatus");

    updateSystemStatus("Indexing...");

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        progressEl.classList.add("active");
        statusEl.textContent = `Ingesting ${file.name}... (${i + 1}/${files.length})`;
        barFill.style.width = "40%";

        const formData = new FormData();
        formData.append("file", file);

        try {
            barFill.style.width = "70%";
            const response = await fetch(apiUrl("/api/upload"), {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Upload failed");
            }

            const data = await response.json();
            barFill.style.width = "100%";
            statusEl.textContent = `✓ Indexed ${data.filename}`;
            showToast(data.message, "success");

        } catch (err) {
            statusEl.textContent = `✗ ${err.message}`;
            showToast(`Failed to upload ${file.name}: ${err.message}`, "error");
        }
    }

    setTimeout(() => {
        progressEl.classList.remove("active");
        barFill.style.width = "0%";
        updateSystemStatus("System Ready");
    }, 1800);

    loadDocuments();
}

async function ingestUrl() {
    const urlInput = document.getElementById("urlInput");
    const ingestBtn = document.getElementById("ingestUrlBtn");
    const progressEl = document.getElementById("uploadProgress");
    const barFill = document.getElementById("progressBarFill");
    const statusEl = document.getElementById("uploadStatus");

    const url = urlInput ? urlInput.value.trim() : "";
    if (!url) {
        showToast("Please enter a valid document or web URL", "info");
        return;
    }

    if (ingestBtn) ingestBtn.disabled = true;
    progressEl.classList.add("active");
    statusEl.textContent = `Fetching content from ${url}...`;
    barFill.style.width = "40%";

    updateSystemStatus("Scraping URL...");

    try {
        barFill.style.width = "75%";
        const response = await fetch(apiUrl("/api/ingest-url"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url }),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Failed to ingest URL");
        }

        const data = await response.json();
        barFill.style.width = "100%";
        statusEl.textContent = `✓ Ingested ${data.filename}`;
        showToast(data.message, "success");
        urlInput.value = "";

    } catch (err) {
        statusEl.textContent = `✗ ${err.message}`;
        showToast(`URL Ingestion failed: ${err.message}`, "error");
    } finally {
        if (ingestBtn) ingestBtn.disabled = false;
        setTimeout(() => {
            progressEl.classList.remove("active");
            barFill.style.width = "0%";
            updateSystemStatus("System Ready");
        }, 2000);
        loadDocuments();
    }
}

// ============================================================
// Document List & Search Filter
// ============================================================
async function loadDocuments() {
    try {
        const response = await fetch(apiUrl("/api/documents"));
        const data = await response.json();
        allDocuments = data.documents || [];

        renderDocumentList(allDocuments);

        const countEl = document.getElementById("docCount");
        if (countEl) countEl.textContent = String(allDocuments.length);

    } catch (err) {
        console.error("Failed to load documents:", err);
    }
}

function renderDocumentList(docs) {
    const listEl = document.getElementById("documentList");
    if (!listEl) return;

    if (docs.length === 0) {
        listEl.innerHTML = `
            <div class="no-documents">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><polyline points="13 2 13 9 20 9"/></svg>
                <span>No matching documents</span>
            </div>`;
        return;
    }

    listEl.innerHTML = docs
        .map((doc) => {
            const ext = doc.file_type.replace(".", "");
            return `<div class="document-item">
                <div class="doc-file-icon ${ext}">${ext.toUpperCase().slice(0, 3)}</div>
                <div class="doc-details">
                    <div class="doc-filename" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
                    <div class="doc-meta">${doc.chunk_count} chunks</div>
                </div>
                <button class="btn-doc-delete" onclick="deleteDocument('${doc.doc_id}', '${escapeHtml(doc.filename)}')" title="Remove document">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                </button>
            </div>`;
        })
        .join("");
}

function filterDocumentList() {
    const query = document.getElementById("docSearchInput").value.toLowerCase().trim();
    if (!query) {
        renderDocumentList(allDocuments);
        return;
    }
    const filtered = allDocuments.filter(d => d.filename.toLowerCase().includes(query));
    renderDocumentList(filtered);
}

async function deleteDocument(docId, filename) {
    if (!confirm(`Delete "${filename}" and its vector index?`)) return;

    try {
        const response = await fetch(apiUrl(`/api/documents/${docId}`), { method: "DELETE" });
        if (!response.ok) throw new Error("Delete failed");

        showToast(`Removed "${filename}"`, "info");
        loadDocuments();
    } catch (err) {
        showToast(`Failed: ${err.message}`, "error");
    }
}

// ============================================================
// Settings Management
// ============================================================
async function loadSettings() {
    try {
        const response = await fetch(apiUrl("/api/settings"));
        const data = await response.json();

        const statusEl = document.getElementById("apiKeyStatus");
        const hfStatusEl = document.getElementById("hfApiKeyStatus");
        const modelSelect = document.getElementById("modelSelect");
        const topKSelect = document.getElementById("topKSelect");
        const activeModelName = document.getElementById("activeModelName");

        if (statusEl) {
            if (data.has_api_key) {
                statusEl.textContent = "● Connected";
                statusEl.className = "api-key-status connected";
            } else {
                statusEl.textContent = "● Key Pending";
                statusEl.className = "api-key-status missing";
            }
        }

        if (hfStatusEl) {
            if (data.has_hf_api_key) {
                hfStatusEl.textContent = "● Connected";
                hfStatusEl.className = "api-key-status connected";
            } else {
                hfStatusEl.textContent = "● Key Pending";
                hfStatusEl.className = "api-key-status missing";
            }
        }

        if (modelSelect) modelSelect.value = data.model;
        if (topKSelect) topKSelect.value = String(data.top_k);

        if (activeModelName) {
            const friendlyNames = {
                "llama-3.3-70b-versatile": "Llama 3.3 70B",
                "llama-3.1-8b-instant": "Llama 3.1 8B",
                "mixtral-8x7b-32768": "Mixtral 8x7B",
                "gemma2-9b-it": "Gemma 2 9B",
            };
            activeModelName.textContent = friendlyNames[data.model] || data.model;
        }
    } catch (err) {
        console.error("Failed to load settings:", err);
    }
}

async function saveSettings() {
    const model = document.getElementById("modelSelect").value;
    const topK = parseInt(document.getElementById("topKSelect").value);

    const body = { model, top_k: topK };
    try {
        const response = await fetch(apiUrl("/api/settings"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });

        if (!response.ok) throw new Error("Failed to save settings");

        const data = await response.json();
        const statusEl = document.getElementById("apiKeyStatus");
        const hfStatusEl = document.getElementById("hfApiKeyStatus");

        if (data.has_api_key && statusEl) {
            statusEl.textContent = "● Connected";
            statusEl.className = "api-key-status connected";
        }

        if (data.has_hf_api_key && hfStatusEl) {
            hfStatusEl.textContent = "● Connected";
            hfStatusEl.className = "api-key-status connected";
        }

        showToast("Preferences saved successfully", "success");
        loadSettings();

    } catch (err) {
        showToast(`Error: ${err.message}`, "error");
    }
}

// ============================================================
// UI Accordion & Mobile Sidebar Controls
// ============================================================
function toggleSection(header) {
    const body = header.nextElementSibling;
    const icon = header.querySelector(".toggle-icon");
    if (!body || !icon) return;

    if (body.classList.contains("collapsed")) {
        body.classList.remove("collapsed");
        icon.classList.add("open");
    } else {
        body.classList.add("collapsed");
        icon.classList.remove("open");
    }
}

function toggleSidebar() {
    const sidebar = document.getElementById("sidebar");
    const overlay = document.getElementById("sidebarOverlay");
    if (sidebar) sidebar.classList.toggle("open");
    if (overlay) overlay.classList.toggle("active");
}

document.getElementById("sidebarOverlay")?.addEventListener("click", toggleSidebar);

// ============================================================
// Toast System
// ============================================================
function showToast(message, type = "info") {
    const container = document.getElementById("toastContainer");
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateX(20px)";
        toast.style.transition = "all 0.3s ease";
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Helper: Escape HTML
function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}
