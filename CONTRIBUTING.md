# Contributing to mlx-mhc

First MLX implementation of DeepSeek's mHC paper. Contributions welcome!

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/mlx-mhc.git
cd mlx-mhc
pip install -e ".[dev]"
pytest
```

## How to Contribute

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Areas for Contribution

- [ ] Benchmarks comparing mHC vs standard residuals
- [ ] Integration examples with mlx-lm
- [ ] Training loop utilities
- [ ] More Sinkhorn variants (unbalanced, partial)
- [ ] Performance optimizations with mx.compile()

## Code Style

- Follow existing patterns in the codebase
- Add tests for new features
- Update docstrings

## Questions?

Open an issue!
