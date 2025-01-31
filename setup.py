from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="customer-insight-api",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A robust API for customer analytics and engagement using AI and ML technologies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/customer-insight-api",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.68.0,<0.69.0",
        "uvicorn>=0.15.0,<0.16.0",
        "sqlalchemy>=1.4.23,<1.5.0",
        "alembic>=1.7.1,<1.8.0",
        "psycopg2-binary>=2.9.1,<2.10.0",
        "pydantic>=1.8.2,<1.9.0",
        "python-jose>=3.3.0,<3.4.0",
        "passlib>=1.7.4,<1.8.0",
        "bcrypt>=3.2.0,<3.3.0",
        "python-multipart>=0.0.5,<0.1.0",
        "aiohttp>=3.7.4,<3.8.0",
        "asyncpg>=0.24.0,<0.25.0",
        "redis>=3.5.3,<3.6.0",
        "celery>=5.1.2,<5.2.0",
        "flower>=1.0.0,<1.1.0",
        "pandas>=1.3.3,<1.4.0",
        "numpy>=1.21.2,<1.22.0",
        "scikit-learn>=0.24.2,<0.25.0",
        "tensorflow>=2.6.0,<2.7.0",
        "torch>=1.9.0,<1.10.0",
        "transformers>=4.10.2,<4.11.0",
        "spacy>=3.1.3,<3.2.0",
        "nltk>=3.6.3,<3.7.0",
        "gensim>=4.1.2,<4.2.0",
        "matplotlib>=3.4.3,<3.5.0",
        "seaborn>=0.11.2,<0.12.0",
        "plotly>=5.3.1,<5.4.0",
        "dash>=2.0.0,<2.1.0",
        "gunicorn>=20.1.0,<20.2.0",
        "supervisor>=4.2.2,<4.3.0",
        "prometheus-client>=0.11.0,<0.12.0",
        "sentry-sdk>=1.4.3,<1.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5,<6.3.0",
            "pytest-asyncio>=0.15.1,<0.16.0",
            "httpx>=0.19.0,<0.20.0",
            "coverage>=5.5,<5.6",
            "black>=21.9b0,<22.0",
            "isort>=5.9.3,<5.10.0",
            "flake8>=3.9.2,<3.10.0",
            "mypy>=0.910,<0.920",
            "sqlalchemy-stubs>=0.4,<0.5",
            "types-redis>=3.5.9,<3.6.0",
            "types-requests>=2.25.9,<2.26.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "customer-insight-api=app.main:main",
        ],
    },
)
