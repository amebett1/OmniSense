/* ===================================================================
   OmniSense — Application Logic (Connected to Backend API)
   =================================================================== */

// ── Config ─────────────────────────────────────────────────────
const API_BASE = window.location.origin + '/api';

// ── State ──────────────────────────────────────────────────────
const state = {
    registeredUsers: [],
    selectedPhotos: [],
    isCameraRunning: false,
    settings: null,
    statusPollTimer: null,
};

// ── DOM Helpers ────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ── API Helper ─────────────────────────────────────────────────
async function api(endpoint, options = {}) {
    try {
        const res = await fetch(`${API_BASE}${endpoint}`, options);
        if (!res.ok) {
            const err = await res.json().catch(() => ({ error: res.statusText }));
            throw new Error(err.error || `HTTP ${res.status}`);
        }
        return await res.json();
    } catch (e) {
        console.error(`API Error [${endpoint}]:`, e);
        throw e;
    }
}

// ── Navigation ─────────────────────────────────────────────────
function initNavigation() {
    $$('.nav-btn').forEach((btn) => {
        btn.addEventListener('click', () => {
            const sectionId = btn.dataset.section;
            $$('.nav-btn').forEach((b) => b.classList.remove('active'));
            btn.classList.add('active');
            $$('.section').forEach((s) => s.classList.remove('section--active'));
            $(`#section-${sectionId}`).classList.add('section--active');

            // Refresh data when switching to list
            if (sectionId === 'list') fetchUsers();
        });
    });
}

// ── Toast ──────────────────────────────────────────────────────
function showToast(message, type = 'info') {
    const container = $('#toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast--${type}`;
    const icons = { success: '✓', error: '✗', info: 'ℹ' };
    toast.innerHTML = `<strong>${icons[type] || 'ℹ'}</strong> ${message}`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(12px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ── Camera ─────────────────────────────────────────────────────
function initCamera() {
    const btnStart = $('#btn-start-camera');
    const btnStop = $('#btn-stop-camera');
    const btnScreenshot = $('#btn-screenshot');
    const videoEl = $('#camera-video');
    const placeholder = $('#camera-placeholder');
    const hudLive = $('#hud-live');

    btnStart.addEventListener('click', async () => {
        try {
            btnStart.disabled = true;
            showToast('Đang khởi động camera...', 'info');

            await api('/camera/start', { method: 'POST' });

            // Chuyển video element thành img để hiển thị MJPEG stream
            const mjpegImg = document.createElement('img');
            mjpegImg.id = 'camera-mjpeg';
            mjpegImg.src = `${API_BASE}/video_feed`;
            mjpegImg.style.cssText = 'width:100%;height:100%;object-fit:cover;display:block;';
            mjpegImg.alt = 'Camera Feed';

            videoEl.style.display = 'none';
            videoEl.parentNode.insertBefore(mjpegImg, videoEl.nextSibling);

            state.isCameraRunning = true;
            placeholder.style.display = 'none';
            hudLive.style.display = 'flex';
            btnStop.disabled = false;

            // Bắt đầu poll trạng thái camera
            startCameraStatusPoll();

            showToast('Camera đã khởi động với nhận diện', 'success');
        } catch (err) {
            btnStart.disabled = false;
            showToast('Lỗi khởi động camera: ' + err.message, 'error');
        }
    });

    btnStop.addEventListener('click', async () => {
        try {
            await api('/camera/stop', { method: 'POST' });
            stopCamera();
            showToast('Camera đã dừng', 'info');
        } catch (err) {
            showToast('Lỗi dừng camera: ' + err.message, 'error');
        }
    });

    btnScreenshot.addEventListener('click', () => {
        if (!state.isCameraRunning) {
            showToast('Hãy khởi động camera trước', 'error');
            return;
        }
        const mjpegImg = $('#camera-mjpeg');
        if (!mjpegImg) return;

        const canvas = document.createElement('canvas');
        canvas.width = mjpegImg.naturalWidth || 1280;
        canvas.height = mjpegImg.naturalHeight || 720;
        canvas.getContext('2d').drawImage(mjpegImg, 0, 0);

        const link = document.createElement('a');
        link.download = `omnisense_${Date.now()}.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();
        showToast('Đã chụp ảnh màn hình', 'success');
    });
}

function stopCamera() {
    state.isCameraRunning = false;
    const mjpegImg = $('#camera-mjpeg');
    if (mjpegImg) {
        mjpegImg.src = '';
        mjpegImg.remove();
    }
    const videoEl = $('#camera-video');
    videoEl.style.display = 'block';
    $('#camera-placeholder').style.display = 'flex';
    $('#hud-live').style.display = 'none';
    $('#btn-start-camera').disabled = false;
    $('#btn-stop-camera').disabled = true;

    // Dừng poll
    if (state.statusPollTimer) {
        clearInterval(state.statusPollTimer);
        state.statusPollTimer = null;
    }

    // Reset stats
    $('#stat-faces').textContent = '0';
    $('#stat-recognized').textContent = '0';
    $('#stat-unknown').textContent = '0';
    $('#stat-inference').textContent = '-- ms';
    $('#hud-fps').textContent = 'FPS: --';
}

function startCameraStatusPoll() {
    if (state.statusPollTimer) clearInterval(state.statusPollTimer);

    state.statusPollTimer = setInterval(async () => {
        if (!state.isCameraRunning) return;
        try {
            const status = await api('/camera/status');
            $('#hud-fps').textContent = `FPS: ${status.fps}`;
            $('#stat-inference').textContent = `${status.inference_ms} ms`;

            const faces = status.faces || [];
            const recognized = faces.filter((f) => f.label !== 'Unknown').length;
            const unknown = faces.filter((f) => f.label === 'Unknown').length;

            $('#stat-faces').textContent = faces.length;
            $('#stat-recognized').textContent = recognized;
            $('#stat-unknown').textContent = unknown;

            // Cập nhật recognition log
            updateRecognitionLog(faces);
        } catch (e) {
            // Ignore polling errors
        }
    }, 1000);
}

function updateRecognitionLog(faces) {
    const logContainer = $('#recognition-log');
    if (!faces || faces.length === 0) return;

    const now = new Date().toLocaleTimeString('vi-VN');

    for (const face of faces) {
        if (face.label === 'Unknown') continue;

        // Tránh duplicate gần nhau
        const existing = logContainer.querySelector(`[data-label="${face.label}"]`);
        if (existing) {
            existing.querySelector('.log-time').textContent = now;
            existing.querySelector('.log-score').textContent = `(${face.score})`;
            continue;
        }

        // Nếu phát hiện người mới -> Tự động gọi LLM sinh câu chào giọng nói
        if (!existing) {
            triggerAutoGreeting(face.label);
        }

        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.dataset.label = face.label;
        entry.innerHTML = `
            <span class="log-dot log-dot--success"></span>
            <span class="log-name">${escapeHtml(face.label)}</span>
            <span class="log-score">(${face.score})</span>
            <span class="log-time">${now}</span>
        `;

        // Xoá placeholder nếu có
        const empty = logContainer.querySelector('.log-empty');
        if (empty) empty.remove();

        logContainer.prepend(entry);

        // Giới hạn 20 entries
        while (logContainer.children.length > 20) {
            logContainer.lastChild.remove();
        }
    }
}

// ── Auto Greeting Trigger ─────────────────────────────────────
let lastGreetedUser = null;
let lastGreetTime = 0;

async function triggerAutoGreeting(label) {
    const now = Date.now();
    // Tránh phát câu chào lặp lại cho cùng 1 người trong vòng 30 giây
    if (lastGreetedUser === label && (now - lastGreetTime) < 30000) return;

    lastGreetedUser = label;
    lastGreetTime = now;

    try {
        const user = state.registeredUsers.find(u => u.id === label || u.name === label);
        const name = user ? user.name : label;
        const role = user ? user.role : 'khác';

        console.log(`✨ Triggering Auto Greeting for ${name} (${role})...`);

        const data = await api('/greet', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, role })
        });

        if (data.greeting_text) {
            // Thêm câu chào vào Chat Box ở Sidebar
            const camMessagesContainer = $('#cam-chat-messages');
            if (camMessagesContainer) {
                const welcomeMsg = camMessagesContainer.querySelector('.chat-welcome-msg');
                if (welcomeMsg) welcomeMsg.remove();

                const bubble = document.createElement('div');
                bubble.className = 'chat-bubble chat-bubble--bot';
                bubble.innerHTML = `
                    <div class="chat-bubble-sender">Trợ lý AI</div>
                    <div class="chat-bubble-text">${escapeHtml(data.greeting_text)}</div>
                `;
                camMessagesContainer.appendChild(bubble);
                camMessagesContainer.scrollTop = camMessagesContainer.scrollHeight;
            }

            // Tự động phát âm thanh câu chào
            if (data.audio_url) {
                const cacheBustUrl = `${window.location.origin}${data.audio_url}?t=${Date.now()}`;
                const audio = new Audio(cacheBustUrl);
                audio.play().catch(e => console.warn('Auto greeting audio playback blocked by browser:', e));
            }
        }
    } catch (e) {
        console.warn('Auto greeting error:', e);
    }
}


// ── Registration ───────────────────────────────────────────────
function initRegistration() {
    const form = $('#register-form');
    const photoInput = $('#reg-photo');
    const dropzone = $('#photo-dropzone');
    const previewList = $('#photo-preview-list');
    const previewName = $('#preview-name');
    const previewBadge = $('#preview-badge');
    const previewGender = $('#preview-gender');
    const previewAvatar = $('#preview-avatar');

    // Live preview
    $('#reg-name').addEventListener('input', (e) => {
        previewName.textContent = e.target.value || 'Họ và Tên';
    });

    $$('input[name="role"]').forEach((radio) => {
        radio.addEventListener('change', (e) => {
            const labels = { lecturer: 'Giảng viên', student: 'Sinh viên', other: 'Khác' };
            previewBadge.textContent = labels[e.target.value] || 'Chức vụ';
        });
    });

    $$('input[name="gender"]').forEach((radio) => {
        radio.addEventListener('change', (e) => {
            const labels = { male: 'Nam', female: 'Nữ' };
            previewGender.textContent = labels[e.target.value] || 'Giới tính';
        });
    });

    // Photo upload
    dropzone.addEventListener('click', () => photoInput.click());
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });
    photoInput.addEventListener('change', (e) => handleFiles(e.target.files));

    function handleFiles(files) {
        const allowed = ['image/jpeg', 'image/png', 'image/bmp', 'image/webp'];
        for (const file of files) {
            if (state.selectedPhotos.length >= 5) {
                showToast('Tối đa 5 ảnh', 'error');
                break;
            }
            if (!allowed.includes(file.type)) {
                showToast(`File "${file.name}" không hỗ trợ`, 'error');
                continue;
            }
            state.selectedPhotos.push(file);
            renderPhotoPreview(file, state.selectedPhotos.length - 1);
        }
    }

    function renderPhotoPreview(file, index) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const item = document.createElement('div');
            item.className = 'photo-preview-item';
            item.dataset.index = index;
            item.innerHTML = `
                <img src="${e.target.result}" alt="preview">
                <button class="photo-preview-remove" type="button">&times;</button>
            `;
            item.querySelector('.photo-preview-remove').addEventListener('click', () => {
                state.selectedPhotos[index] = null;
                item.remove();
                updateAvatarPreview();
            });
            previewList.appendChild(item);
            updateAvatarPreview();
        };
        reader.readAsDataURL(file);
    }

    function updateAvatarPreview() {
        const firstPhoto = state.selectedPhotos.find((p) => p !== null);
        if (firstPhoto) {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewAvatar.innerHTML = `<img src="${e.target.result}" alt="avatar">`;
            };
            reader.readAsDataURL(firstPhoto);
        } else {
            previewAvatar.innerHTML = `<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="8" r="4"/><path d="M5 20c0-4 3.5-7 7-7s7 3 7 7"/></svg>`;
        }
    }

    // Submit — gửi API thay vì localStorage
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const name = $('#reg-name').value.trim();
        const roleEl = document.querySelector('input[name="role"]:checked');
        const genderEl = document.querySelector('input[name="gender"]:checked');
        const photos = state.selectedPhotos.filter((p) => p !== null);

        if (!name) return showToast('Vui lòng nhập họ và tên', 'error');
        if (!roleEl) return showToast('Vui lòng chọn chức vụ', 'error');
        if (!genderEl) return showToast('Vui lòng chọn giới tính', 'error');
        if (photos.length === 0) return showToast('Vui lòng thêm ít nhất 1 ảnh', 'error');

        // Tạo FormData để gửi file
        const formData = new FormData();
        formData.append('name', name);
        formData.append('role', roleEl.value);
        formData.append('gender', genderEl.value);
        photos.forEach((photo) => formData.append('photos', photo));

        const btnSubmit = $('#btn-register');
        btnSubmit.disabled = true;
        btnSubmit.textContent = 'Đang đăng ký...';

        try {
            const result = await fetch(`${API_BASE}/users`, {
                method: 'POST',
                body: formData,
            });
            const data = await result.json();

            if (!result.ok) throw new Error(data.error);

            showToast(`Đã đăng ký thành công: ${name} (${data.photo_count} ảnh)`, 'success');
            resetForm();
        } catch (err) {
            showToast('Lỗi đăng ký: ' + err.message, 'error');
        } finally {
            btnSubmit.disabled = false;
            btnSubmit.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="8.5" cy="7" r="4"/><line x1="20" y1="8" x2="20" y2="14"/><line x1="23" y1="11" x2="17" y2="11"/></svg>
                Đăng Ký`;
        }
    });

    function resetForm() {
        form.reset();
        state.selectedPhotos = [];
        previewList.innerHTML = '';
        previewName.textContent = 'Họ và Tên';
        previewBadge.textContent = 'Chức vụ';
        previewGender.textContent = 'Giới tính';
        previewAvatar.innerHTML = `<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="8" r="4"/><path d="M5 20c0-4 3.5-7 7-7s7 3 7 7"/></svg>`;
    }

    $('#btn-reset-form').addEventListener('click', resetForm);
}

// ── User List ──────────────────────────────────────────────────
async function fetchUsers() {
    try {
        state.registeredUsers = await api('/users');
        renderUserList();
    } catch (err) {
        showToast('Lỗi tải danh sách: ' + err.message, 'error');
    }
}

function renderUserList() {
    const grid = $('#user-grid');
    const searchTerm = ($('#search-input')?.value || '').toLowerCase();
    const filterRole = $('#filter-role')?.value || 'all';
    const filterGender = $('#filter-gender')?.value || 'all';

    let filtered = state.registeredUsers.filter((u) => {
        const matchName = u.name.toLowerCase().includes(searchTerm);
        const matchRole = filterRole === 'all' || u.role === filterRole;
        const matchGender = filterGender === 'all' || u.gender === filterGender;
        return matchName && matchRole && matchGender;
    });

    $('#list-count').textContent = `${filtered.length} người đã đăng ký`;

    if (filtered.length === 0) {
        grid.innerHTML = `
            <div class="empty-state">
                <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
                    <circle cx="9" cy="7" r="4"/>
                    <path d="M23 21v-2a4 4 0 0 0-3-3.87"/>
                    <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
                </svg>
                <h3>${searchTerm ? 'Không tìm thấy kết quả' : 'Chưa có ai đăng ký'}</h3>
                <p>${searchTerm ? 'Thử thay đổi từ khoá tìm kiếm' : 'Hãy thêm khuôn mặt mới từ tab "Đăng Ký"'}</p>
            </div>`;
        return;
    }

    const roleLabels = { lecturer: 'Giảng viên', student: 'Sinh viên', other: 'Khác' };
    const genderLabels = { male: 'Nam', female: 'Nữ' };

    grid.innerHTML = filtered.map((u) => {
        // Lấy ảnh đầu tiên từ database nếu có
        const photoSrc = u.photos && u.photos.length > 0
            ? `${API_BASE}/users/${u.id}/photo/${u.photos[0]}`
            : '';
        const avatarContent = photoSrc
            ? `<img src="${photoSrc}" alt="${escapeHtml(u.name)}">`
            : getInitials(u.name);

        return `
        <div class="user-card" data-id="${u.id}">
            <div class="user-card-avatar">${avatarContent}</div>
            <div class="user-card-info">
                <div class="user-card-name">${escapeHtml(u.name)}</div>
                <div class="user-card-meta">
                    <span class="meta-tag meta-tag--role">${roleLabels[u.role] || u.role}</span>
                    <span class="meta-tag meta-tag--gender">${genderLabels[u.gender] || u.gender}</span>
                    <span class="meta-tag" style="background:rgba(255,255,255,0.06);color:var(--text-muted);">${u.photo_count} ảnh</span>
                </div>
            </div>
            <div class="user-card-actions">
                <button class="btn btn--secondary btn--sm" onclick="openEditModal('${u.id}')" title="Chỉnh sửa">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                        <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                    </svg>
                </button>
                <button class="btn btn--danger btn--sm" onclick="deleteUser('${u.id}')" title="Xoá">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="3 6 5 6 21 6"/>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                    </svg>
                </button>
            </div>
        </div>`;
    }).join('');
}

function getInitials(name) {
    return name.split(' ').map((w) => w[0]).join('').substring(0, 2).toUpperCase();
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

async function deleteUser(id) {
    if (!confirm('Bạn có chắc muốn xoá người này?\nẢnh trong database cũng sẽ bị xoá!')) return;
    try {
        await api(`/users/${id}`, { method: 'DELETE' });
        showToast('Đã xoá thành công', 'success');
        fetchUsers();
    } catch (err) {
        showToast('Lỗi xoá: ' + err.message, 'error');
    }
}

// ── Edit Modal ─────────────────────────────────────────────────
function openEditModal(id) {
    const user = state.registeredUsers.find((u) => u.id === id);
    if (!user) return;

    $('#edit-id').value = id;
    $('#edit-name').value = user.name;
    $('#edit-role').value = user.role || 'other';
    $('#edit-gender').value = user.gender;

    renderEditPhotosList(user);
    $('#edit-modal').style.display = 'grid';
}

function renderEditPhotosList(user) {
    const container = $('#edit-photos-list');
    if (!container) return;

    if (!user.photos || user.photos.length === 0) {
        container.innerHTML = `<span class="edit-photo-grid-empty">Chưa có ảnh nào được đăng ký</span>`;
        return;
    }

    container.innerHTML = user.photos.map((photo) => {
        const photoUrl = `${API_BASE}/users/${user.id}/photo/${photo}`;
        return `
            <div class="edit-photo-item" data-photo="${escapeHtml(photo)}">
                <img src="${photoUrl}" alt="User Photo">
                <button type="button" class="edit-photo-del" title="Xoá ảnh" onclick="deleteUserPhoto('${user.id}', '${escapeHtml(photo)}')">&times;</button>
            </div>
        `;
    }).join('');
}

async function deleteUserPhoto(userId, photoName) {
    if (!confirm('Bạn có chắc muốn xoá ảnh này khỏi hệ thống?')) return;

    try {
        const res = await api(`/users/${userId}/photos/${photoName}`, { method: 'DELETE' });
        showToast('Đã xoá ảnh thành công', 'success');

        // Cập nhật lại user trong state và UI
        const user = state.registeredUsers.find((u) => u.id === userId);
        if (user) {
            user.photos = res.photos;
            user.photo_count = res.photos.length;
            renderEditPhotosList(user);
        }
        fetchUsers();
    } catch (err) {
        showToast('Lỗi xoá ảnh: ' + err.message, 'error');
    }
}

function initModal() {
    const modal = $('#edit-modal');
    const form = $('#edit-form');

    $('#modal-close').addEventListener('click', () => (modal.style.display = 'none'));
    $('#modal-cancel').addEventListener('click', () => (modal.style.display = 'none'));
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.style.display = 'none';
    });

    const triggerBtn = $('#btn-trigger-add-photo');
    const photoInput = $('#edit-photo-input');

    if (triggerBtn && photoInput) {
        triggerBtn.addEventListener('click', () => photoInput.click());
        photoInput.addEventListener('change', async (e) => {
            const files = Array.from(e.target.files);
            if (!files.length) return;

            const userId = $('#edit-id').value;
            if (!userId) return;

            const formData = new FormData();
            files.forEach((f) => formData.append('photos', f));

            try {
                showToast('Đang tải ảnh mới lên...', 'info');
                const res = await fetch(`${API_BASE}/users/${userId}/photos`, {
                    method: 'POST',
                    body: formData,
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || 'Upload failed');

                showToast(`Đã thêm ${data.added} ảnh mới`, 'success');
                photoInput.value = '';

                // Cập nhật state & UI
                const user = state.registeredUsers.find((u) => u.id === userId);
                if (user) {
                    user.photos = data.photos;
                    user.photo_count = data.photos.length;
                    renderEditPhotosList(user);
                }
                fetchUsers();
            } catch (err) {
                showToast('Lỗi tải ảnh: ' + err.message, 'error');
            }
        });
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const id = $('#edit-id').value;
        const data = {
            name: $('#edit-name').value.trim(),
            role: $('#edit-role').value,
            gender: $('#edit-gender').value,
        };

        try {
            await api(`/users/${id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });
            showToast('Đã cập nhật thành công', 'success');
            modal.style.display = 'none';
            fetchUsers();
        } catch (err) {
            showToast('Lỗi cập nhật: ' + err.message, 'error');
        }
    });
}


// ── List Filters ───────────────────────────────────────────────
function initListFilters() {
    $('#search-input').addEventListener('input', renderUserList);
    $('#filter-role').addEventListener('change', renderUserList);
    $('#filter-gender').addEventListener('change', renderUserList);
}

// ── Settings ───────────────────────────────────────────────────
async function initSettings() {
    const thresholdSlider = $('#param-threshold');
    const threadsSlider = $('#param-threads');
    const thresholdValue = $('#threshold-value');
    const threadsValue = $('#threads-value');

    // Tải cài đặt từ backend
    try {
        state.settings = await api('/settings');
        loadSettingsUI();
    } catch (e) {
        console.warn('Không tải được settings từ backend, dùng mặc định');
    }

    thresholdSlider.addEventListener('input', (e) => {
        thresholdValue.textContent = parseFloat(e.target.value).toFixed(2);
    });

    threadsSlider.addEventListener('input', (e) => {
        threadsValue.textContent = e.target.value;
    });

    $$('input[name="model"]').forEach((radio) => {
        radio.addEventListener('change', () => {
            $$('.model-option').forEach((opt) => opt.classList.remove('model-option--active'));
            radio.closest('.model-option').classList.add('model-option--active');
            const recommended = radio.value === 'buffalo_sc' ? 0.40 : 0.50;
            thresholdSlider.value = recommended;
            thresholdValue.textContent = recommended.toFixed(2);
        });
    });

    // Lưu — gửi API
    $('#btn-save-settings').addEventListener('click', async () => {
        const settings = {
            model: document.querySelector('input[name="model"]:checked').value,
            threshold: parseFloat(thresholdSlider.value),
            threads: parseInt(threadsSlider.value),
            det_size: parseInt($('#param-det-size').value),
            camera_index: parseInt($('#param-camera-index').value),
            cuda: $('#toggle-cuda').checked,
        };

        try {
            await api('/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings),
            });
            state.settings = settings;
            showToast('Đã lưu cài đặt (cần khởi động lại server để áp dụng model mới)', 'success');
        } catch (err) {
            showToast('Lỗi lưu cài đặt: ' + err.message, 'error');
        }
    });

    // Reset
    $('#btn-reset-settings').addEventListener('click', async () => {
        const defaults = {
            model: 'buffalo_sc',
            threshold: 0.40,
            threads: 8,
            det_size: 640,
            camera_index: 0,
            cuda: false,
        };

        try {
            await api('/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(defaults),
            });
            state.settings = defaults;
            loadSettingsUI();
            showToast('Đã khôi phục cài đặt mặc định', 'info');
        } catch (err) {
            showToast('Lỗi reset: ' + err.message, 'error');
        }
    });
}

function loadSettingsUI() {
    const s = state.settings;
    if (!s) return;

    const modelRadio = document.querySelector(`input[name="model"][value="${s.model}"]`);
    if (modelRadio) {
        modelRadio.checked = true;
        $$('.model-option').forEach((opt) => opt.classList.remove('model-option--active'));
        modelRadio.closest('.model-option').classList.add('model-option--active');
    }

    $('#param-threshold').value = s.threshold;
    $('#threshold-value').textContent = s.threshold.toFixed(2);
    $('#param-threads').value = s.threads;
    $('#threads-value').textContent = s.threads;
    $('#param-det-size').value = s.det_size;
    $('#param-camera-index').value = s.camera_index;
    $('#toggle-cuda').checked = s.cuda;
}

// ── Model Status Indicator ─────────────────────────────────────
async function checkModelStatus() {
    try {
        const status = await api('/status');
        if (status.model_ready) {
            showToast(`Model ${status.model_name} sẵn sàng — ${status.database_count} người trong DB`, 'success');
        } else {
            showToast('Model đang khởi tạo, vui lòng đợi...', 'info');
            // Retry sau 5 giây
            setTimeout(checkModelStatus, 5000);
        }
    } catch (e) {
        showToast('Không kết nối được backend API. Hãy chạy: python backend/app.py', 'error');
    }
}

// ── Voice Assistant ───────────────────────────────────────────
function initVoiceAssistant() {
    const micBtn = $('#voice-mic-btn');
    const statusText = $('#voice-status-text');
    const userTranscriptElem = $('#voice-user-transcript');
    const botReplyElem = $('#voice-bot-reply');

    // Elements cho Camera Sidebar Chat Box
    const camMicBtn = $('#cam-voice-mic-btn');
    const camStatusBadge = $('#cam-voice-status');
    const camMessagesContainer = $('#cam-chat-messages');
    const camChatInput = $('#cam-chat-input');
    const camChatSendBtn = $('#cam-chat-send-btn');

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition = null;

    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.lang = 'vi-VN';
        recognition.continuous = true;
        recognition.interimResults = false;
    } else {
        if (statusText) {
            statusText.textContent = "Trình duyệt không hỗ trợ Web Speech API!";
            statusText.style.color = "#ef4444";
        }
        if (camStatusBadge) {
            camStatusBadge.textContent = "STT Không hỗ trợ";
            camStatusBadge.style.color = "#ef4444";
        }
        if (micBtn) micBtn.disabled = true;
        if (camMicBtn) camMicBtn.disabled = true;
    }

    let isListening = false;
    let isSpeaking = false;

    // Helper cập nhật Trạng thái giao diện
    function updateStatusUI(text, stateType = 'idle') {
        if (statusText) {
            statusText.textContent = text;
            const colors = { listening: '#ef4444', thinking: '#f59e0b', speaking: '#10b981', idle: '#38bdf8' };
            statusText.style.color = colors[stateType] || '#38bdf8';
        }
        if (camStatusBadge) {
            camStatusBadge.textContent = text;
            camStatusBadge.className = 'chat-status-badge';
            if (stateType === 'listening') camStatusBadge.classList.add('listening');
            else if (stateType === 'speaking') camStatusBadge.classList.add('speaking');
        }
    }

    function updateMicButtons(listening) {
        if (micBtn) {
            micBtn.style.background = listening ? '#ef4444' : '#334155';
            micBtn.style.borderColor = listening ? '#fca5a5' : 'var(--accent-color, #38bdf8)';
            if (listening) micBtn.classList.add('listening');
            else micBtn.classList.remove('listening');
        }
        if (camMicBtn) {
            camMicBtn.style.background = listening ? '#ef4444' : 'var(--bg-glass)';
            camMicBtn.style.borderColor = listening ? '#fca5a5' : 'var(--border)';
            if (listening) camMicBtn.classList.add('listening');
            else camMicBtn.classList.remove('listening');
        }
    }

    function startListening() {
        if (!recognition || isSpeaking) return;
        try {
            recognition.start();
            isListening = true;
            updateMicButtons(true);
            updateStatusUI('Đang lắng nghe...', 'listening');
        } catch (e) {
            console.warn('SpeechRecognition start error:', e);
        }
    }

    function stopListening() {
        if (!recognition) return;
        try {
            recognition.stop();
            isListening = false;
            updateMicButtons(false);
            updateStatusUI('Sẵn sàng', 'idle');
        } catch (e) {
            console.warn('SpeechRecognition stop error:', e);
        }
    }

    function toggleListening() {
        if (isSpeaking) return;
        if (isListening) stopListening();
        else startListening();
    }

    if (micBtn) micBtn.addEventListener('click', toggleListening);
    if (camMicBtn) camMicBtn.addEventListener('click', toggleListening);

    // Xử lý Thêm bong bóng Chat vào Chat Box ở Camera Sidebar
    function appendChatBubble(sender, text) {
        if (!camMessagesContainer) return;

        // Xoá welcome message nếu còn
        const welcomeMsg = camMessagesContainer.querySelector('.chat-welcome-msg');
        if (welcomeMsg) welcomeMsg.remove();

        const bubble = document.createElement('div');
        bubble.className = `chat-bubble chat-bubble--${sender}`;
        
        const senderLabel = sender === 'user' ? 'Bạn' : 'Trợ lý AI';
        bubble.innerHTML = `
            <div class="chat-bubble-sender">${senderLabel}</div>
            <div class="chat-bubble-text">${escapeHtml(text)}</div>
        `;

        camMessagesContainer.appendChild(bubble);
        camMessagesContainer.scrollTop = camMessagesContainer.scrollHeight;
    }

    // Gửi chat text khi nhập từ input box
    async function handleSendText() {
        if (!camChatInput) return;
        const text = camChatInput.value.trim();
        if (!text) return;

        camChatInput.value = '';
        await processUserUtterance(text);
    }

    if (camChatSendBtn) camChatSendBtn.addEventListener('click', handleSendText);
    if (camChatInput) {
        camChatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') handleSendText();
        });
    }

    // Xử lý khi nhận giọng nói từ Web Speech API
    if (recognition) {
        recognition.onresult = async (event) => {
            const lastIndex = event.results.length - 1;
            const transcript = event.results[lastIndex][0].transcript.trim();

            if (!transcript) return;
            console.log('🎤 User Speech Transcript:', transcript);
            await processUserUtterance(transcript);
        };

        recognition.onerror = (event) => {
            console.error('Speech Recognition error:', event.error);
            if (event.error === 'not-allowed') {
                updateStatusUI('Vui lòng cho phép truy cập Microphone!', 'idle');
            }
        };
    }

    // Luồng xử lý chung cho câu nói / văn bản người dùng
    async function processUserUtterance(text) {
        // Cập nhật giao diện
        if (userTranscriptElem) userTranscriptElem.textContent = text;
        appendChatBubble('user', text);
        updateStatusUI('Đang suy nghĩ...', 'thinking');

        try {
            const data = await api('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            const replyText = data.reply_text || '...';
            if (botReplyElem) botReplyElem.textContent = replyText;
            appendChatBubble('bot', replyText);

            if (data.audio_url) {
                playResponseAudio(data.audio_url);
            } else {
                updateStatusUI('Hoàn tất', 'idle');
            }
        } catch (err) {
            console.error('❌ Chat API Error:', err);
            if (botReplyElem) botReplyElem.textContent = 'Lỗi kết nối Trợ lý AI!';
            appendChatBubble('bot', 'Có lỗi kết nối tới Server Trợ lý AI.');
            updateStatusUI('Lỗi xử lý!', 'idle');
            showToast('Lỗi Trợ lý AI: ' + err.message, 'error');
        }
    }

    function playResponseAudio(audioUrl) {
        // Tắt micro trước khi phát loa để TRÁNH ECHO
        isSpeaking = true;
        stopListening();

        updateStatusUI('Đang trả lời...', 'speaking');

        // Cache-busting URL
        const cacheBustUrl = `${window.location.origin}${audioUrl}?t=${Date.now()}`;
        const audio = new Audio(cacheBustUrl);

        audio.onended = () => {
            console.log('🔊 Phát audio xong, bật lại Micro.');
            isSpeaking = false;
            startListening();
        };

        audio.onerror = (err) => {
            console.error('❌ Lỗi phát audio:', err);
            isSpeaking = false;
            startListening();
        };

        audio.play().catch((err) => {
            console.warn('⚠️ Autoplay bị chặn:', err);
            isSpeaking = false;
            updateStatusUI('Bấm 🎙️ để tiếp tục', 'idle');
        });
    }
}


// ── Init ───────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initCamera();
    initRegistration();
    initModal();
    initListFilters();
    initSettings();
    initVoiceAssistant();
    fetchUsers();

    // Kiểm tra trạng thái model khi load trang
    setTimeout(checkModelStatus, 1500);
});

