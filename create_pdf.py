from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

text_lines = [
    "Artificial Intelligence and Machine Learning",
    "",
    "Artificial intelligence is the simulation of human intelligence by machines.",
    "Machine learning is a subset of AI that allows systems to learn from data.",
    "Deep learning uses neural networks with many layers to solve complex problems.",
    "Natural language processing enables computers to understand human language.",
    "Large language models like GPT are trained on vast amounts of text data.",
    "These models can generate text, answer questions, and assist with various tasks.",
    "",
    "History of Artificial Intelligence",
    "",
    "The term artificial intelligence was coined by John McCarthy in 1956.",
    "Early AI research focused on problem solving and symbolic methods.",
    "In the 1980s, expert systems became popular in the industry.",
    "The AI winter was a period of reduced funding and interest in AI research.",
    "Neural networks gained popularity in the 1990s with backpropagation.",
    "In 2012, deep learning revolutionized image recognition with AlexNet.",
    "Transformer architecture introduced in 2017 changed NLP forever.",
    "GPT models by OpenAI brought generative AI to mainstream in 2020.",
    "",
    "Applications of AI",
    "",
    "AI is used in healthcare for disease diagnosis and drug discovery.",
    "Self-driving cars use computer vision and reinforcement learning.",
    "Recommendation systems power Netflix, YouTube, and Spotify.",
    "AI assistants like Siri and Alexa use natural language processing.",
    "Fraud detection in banking relies heavily on machine learning models.",
    "",
    "Challenges in AI",
    "",
    "Bias in AI models is a major ethical concern in the industry.",
    "AI systems require large amounts of data to train effectively.",
    "Explainability of AI decisions is difficult in deep learning models.",
    "Privacy concerns arise when AI systems collect personal data.",
    "Energy consumption of large AI models is an environmental concern.",
    "",
    "Future of AI",
    "",
    "Artificial general intelligence aims to match human level reasoning.",
    "Quantum computing may accelerate AI training significantly.",
    "AI regulation is being discussed by governments around the world.",
    "Multimodal AI can process text, images, audio, and video together.",
    "AI agents can autonomously complete complex multi-step tasks.",
    "Human-AI collaboration will define the future of work and creativity.",
]

c = canvas.Canvas("sample2.pdf", pagesize=A4)
width, height = A4
y = height - 50

for line in text_lines:
    if y < 50:
        c.showPage()
        y = height - 50
    c.setFont("Helvetica", 11)
    c.drawString(50, y, line)
    y -= 20

c.save()
print("sample2.pdf created successfully!")