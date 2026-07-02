from server import create_app


app = create_app()


if __name__ == "__main__":
    print("🚀 Server berjalan di PORT 5002...")
    print("\n📡 Available Endpoints:")
    print("   POST   /extract-cv/          - Extract CV and analyze skills")
    print("   POST   /set-groq-key/        - Set Groq API Key")
    print("   GET    /api/skills           - Get all skills")
    print("   POST   /api/skills           - Add new skill")
    print("   DELETE /api/skills?id=X      - Delete skill by ID")
    print("   GET    /api/jobs             - Get all jobs")
    print("   DELETE /api/jobs?id=X        - Delete job by ID")
    print("   POST   /api/check-links      - Check status of all job URLs")
    print("   POST   /api/delete-expired   - Delete all expired jobs")
    app.run(host="127.0.0.1", debug=False, port=5002, use_reloader=False, threaded=True)
