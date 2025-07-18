from locust import HttpUser, task, between

class AIUser(HttpUser):
    host = "http://localhost:8000"
    wait_time = between(1, 5)

    @task
    def generate_text(self):
        self.client.post(
            "/generate",
            json={"prompt": "Performance test prompt", "max_length": 50},
            headers={"Content-Type": "application/json"}
        )
