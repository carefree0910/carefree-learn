![carefree-learn][socialify-image]

Deep Learning with [PyTorch](https://pytorch.org/) made easy 🚀 !

## v0.5.x WIP!

Here are the main design principles:
- The codes should be '`module` first', which means all previous `model`s should be a simple `module` now.
  - And `model` should only be related to the training stuffs. If we only want to use the fancy AI models at inference stage, `module` should be all we need.
- The `module`s should be as 'native' as possible: no inheritance from base classes except `nn.Module` should be the best, and previous inheritance-based features should be achieved by **dependency injection**.
  - This helps the `module`s to be more `torch.compile` friendly.
- Training stuffs are not considered at the first place, but they will definitely be added later on, based on the modern AI developments.
- APIs will be as BC as possible.

## License

`carefree-learn` is MIT licensed, as found in the [`LICENSE`](https://carefree0910.me/carefree-learn-doc/docs/about/license) file.


[socialify-image]: https://socialify.git.ci/carefree0910/carefree-learn/image?description=1&descriptionEditable=Deep%20Learning%20%E2%9D%A4%EF%B8%8F%20PyTorch&forks=1&issues=1&logo=https%3A%2F%2Fraw.githubusercontent.com%2Fcarefree0910%2Fcarefree-learn-doc%2Fmaster%2Fstatic%2Fimg%2Flogo.min.svg&pattern=Floating%20Cogs&stargazers=1