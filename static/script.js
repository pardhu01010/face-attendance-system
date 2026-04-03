document.addEventListener('DOMContentLoaded', () => {
    // ---- TABS LOGIC ----
    const navBtns = document.querySelectorAll('.nav-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active classes
            navBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(t => t.classList.add('hidden'));

            // Add active to clicked
            btn.classList.add('active');
            const target = document.getElementById(btn.dataset.tab);
            target.classList.remove('hidden');

            // If attendance tab, load data
            if (btn.dataset.tab === 'list-tab') {
                loadAttendance();
            }
        });
    });

    // ---- FILE UPLOADS LOGIC ----
    setupFileDrop('authFile', 'authForm');
    setupFileDrop('regFiles', 'regForm', true);

    // Auth Image Preview
    document.getElementById('authFile').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const preview = document.getElementById('authPreview');
            const wrapper = document.getElementById('authPreviewWrapper');
            preview.src = URL.createObjectURL(file);
            wrapper.classList.remove('hidden');
        }
    });

    // Register File List Names
    document.getElementById('regFiles').addEventListener('change', (e) => {
        const files = Array.from(e.target.files);
        const listDiv = document.getElementById('regFilesList');
        if (files.length > 0) {
            listDiv.innerHTML = `<em>Selected ${files.length} images.</em>`;
        } else {
            listDiv.innerHTML = '';
        }
    });

    // ---- API LOGIC ----

    // 1. Authenticate Submit
    document.getElementById('authForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const fileInput = document.getElementById('authFile');
        if (fileInput.files.length === 0) return;

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        await handleApiSubmit('/authenticate/', formData, 'authSubmitBtn', (res) => {
            if (res.status === 'success') {
                showToast(`Verified! Welcome ${res.name}`, 'success');
                // clear
                fileInput.value = '';
                document.getElementById('authPreviewWrapper').classList.add('hidden');
            } else {
                showToast(res.message || 'Face not recognized', 'error');
            }
        });
    });

    // 2. Register Submit
    document.getElementById('regForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const filesInput = document.getElementById('regFiles');
        const nameInput = document.getElementById('regName');
        
        if (filesInput.files.length === 0 || !nameInput.value) return;

        const formData = new FormData();
        formData.append('name', nameInput.value);
        Array.from(filesInput.files).forEach(f => {
            formData.append('files', f);
        });

        await handleApiSubmit('/register/', formData, 'regSubmitBtn', (res) => {
            if (res.status === 'success') {
                showToast(`Successfully registered ${nameInput.value}!`, 'success');
                e.target.reset();
                document.getElementById('regFilesList').innerHTML = '';
            } else {
                showToast(res.message || 'Registration failed', 'error');
            }
        });
    });

    // 3. Load Attendance
    document.getElementById('refreshBtn').addEventListener('click', loadAttendance);

    async function loadAttendance() {
        const today = new Date().toISOString().split('T')[0];
        document.getElementById('currentDateDisplay').innerText = new Date().toDateString();
        
        try {
            const resp = await fetch(`/attendance/?date=${today}`);
            const data = await resp.json();
            
            const tbody = document.getElementById('attendanceBody');
            const emptyState = document.getElementById('emptyState');
            tbody.innerHTML = '';
            
            if (data.attendance && data.attendance.length > 0) {
                emptyState.classList.add('hidden');
                data.attendance.forEach(record => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td><strong>${record.Name}</strong></td>
                        <td><span style="color:var(--text-muted)">${record.Time}</span></td>
                    `;
                    tbody.appendChild(tr);
                });
            } else {
                emptyState.classList.remove('hidden');
            }
        } catch (err) {
            showToast('Failed to load logs', 'error');
        }
    }


    // ---- HELPERS ----
    function setupFileDrop(inputId, formId, multiple = false) {
        const input = document.getElementById(inputId);
        const dropArea = input.closest('.file-drop-area');

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => dropArea.classList.add('is-active'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => dropArea.classList.remove('is-active'), false);
        });

        dropArea.addEventListener('drop', (e) => {
            input.files = e.dataTransfer.files;
            input.dispatchEvent(new Event('change'));
        });
    }

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    async function handleApiSubmit(url, formData, btnId, onSuccess) {
        const btn = document.getElementById(btnId);
        const text = btn.querySelector('.btn-text');
        const spinner = btn.querySelector('.spinner');
        
        btn.disabled = true;
        text.classList.add('hidden');
        spinner.classList.remove('hidden');

        try {
            const resp = await fetch(url, {
                method: 'POST',
                body: formData
            });
            const data = await resp.json();
            
            if (!resp.ok) throw data;
            onSuccess(data);
        } catch (err) {
            showToast(err.detail?.[0]?.msg || err.message || 'An error occurred', 'error');
        } finally {
            btn.disabled = false;
            text.classList.remove('hidden');
            spinner.classList.add('hidden');
        }
    }

    function showToast(msg, type = 'success') {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        // icon
        const icon = type === 'success' ? '✅' : '❌';
        toast.innerHTML = `<span>${icon}</span> <div>${msg}</div>`;
        
        container.appendChild(toast);
        
        setTimeout(() => {
            toast.classList.add('fade-out');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
});
