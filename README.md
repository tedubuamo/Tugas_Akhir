# JobMatcher VPS Setup

Setup ini memisahkan aplikasi menjadi 3 container:

- `backend`: Flask + model inference
- `frontend`: static HTML/CSS/JS
- `caddy`: reverse proxy + HTTPS

Struktur file yang dipakai:

```text
Aplikasi/
├── backend/
│   ├── app.py
│   ├── Dockerfile
│   ├── .env
│   ├── my_db.db
│   └── final_bert_model_update/
├── frontend/
│   ├── Dockerfile
│   ├── index.html
│   ├── analyzer.html
│   ├── admin.html
│   ├── login.html
│   └── public/
├── Caddyfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 1. Kebutuhan VPS

- Ubuntu 22.04 / 24.04
- Docker
- Docker Compose plugin
- Domain yang sudah diarahkan ke IP VPS
- Port `80` dan `443` terbuka

Install Docker:

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Opsional agar tidak perlu `sudo`:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## 2. Upload project ke VPS

Clone repo atau upload folder project ke VPS:

```bash
git clone <repo-url> jobmatcher
cd jobmatcher
```

## 3. Siapkan file environment backend

Buat `backend/.env`:

```env
GROQ_API_KEY=your_groq_api_key
```

## 4. Pastikan file penting ada

Wajib ada:

- `backend/my_db.db`
- `backend/final_bert_model_update/`

Cek cepat:

```bash
ls backend
ls backend/final_bert_model_update
```

## 5. Atur domain untuk Caddy

`Caddyfile` memakai placeholder env:

```caddy
{$APP_DOMAIN} {
    encode gzip zstd

    @backend path /extract-cv/* /set-groq-key/* /api/*
    reverse_proxy @backend backend:5002

    reverse_proxy frontend:80
}
```

Saat menjalankan compose, set `APP_DOMAIN`.

Contoh:

```bash
export APP_DOMAIN=jobmatcher.domainanda.com
```

## 6. Build dan jalankan

```bash
docker compose build
docker compose up -d
```

Cek status:

```bash
docker compose ps
```

Lihat log:

```bash
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f caddy
```

## 7. Akses aplikasi

Jika domain dan DNS sudah benar:

```text
https://jobmatcher.domainanda.com
```

Endpoint penting:

- Frontend: `/`
- Analyzer: `/analyzer`
- Admin: `/admin`
- API jobs: `/api/jobs`
- CV extraction: `/extract-cv/`

## 8. Cara update aplikasi

Jika ada perubahan code:

```bash
git pull
docker compose build
docker compose up -d
```

Jika hanya restart:

```bash
docker compose restart
```

## 9. Backup database

Karena SQLite dipakai langsung dari file host:

```bash
cp backend/my_db.db backend/my_db.db.bak
```

Atau simpan ke folder backup:

```bash
mkdir -p backups
cp backend/my_db.db backups/my_db-$(date +%F-%H%M%S).db
```

## 10. Troubleshooting

### Backend gagal start

```bash
docker compose logs -f backend
```

Penyebab umum:

- `backend/.env` belum ada
- `my_db.db` tidak ada
- folder model `final_bert_model_update` tidak ada
- dependency ML terlalu berat untuk resource VPS

### HTTPS tidak aktif

Pastikan:

- domain mengarah ke IP VPS
- port `80` dan `443` terbuka
- `APP_DOMAIN` sudah di-set sebelum `docker compose up`

### Frontend terbuka tapi API gagal

Cek backend:

```bash
docker compose ps
docker compose logs -f backend
```

Cek response API:

```bash
curl https://jobmatcher.domainanda.com/api/jobs
```

### Inference lambat

Normal jika VPS CPU-only dan model besar. Minimal rekomendasi:

- 4 vCPU
- 8 GB RAM
- SSD storage

## 11. Catatan deployment

- Frontend dipisah ke container sendiri supaya static serving tetap ringan.
- Backend tetap bisa serve frontend secara lokal, tapi pada deployment VPS container frontend dipakai sebagai source utama halaman web.
- Caddy menangani reverse proxy dan TLS otomatis.

## 12. File yang ditambahkan

- [backend/Dockerfile](backend/Dockerfile)
- [frontend/Dockerfile](frontend/Dockerfile)
- [docker-compose.yml](docker-compose.yml)
- [Caddyfile](Caddyfile)
- [README.md](README.md)
