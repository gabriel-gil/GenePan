# GenePan

GenePan is a Python tool for identifying only-tumor and only-normal genes from cancer RNA-seq FPKM data, constructing diagnostic gene panels, and exporting binary matrices suitable for downstream Gene Deregulation Network construction with CChains.

For a selected cancer cohort, GenePan can:

- discover T-gene and N-gene pools across above, below, outside, inside, and mixed family modes;
- construct perfect gene panels with configurable tie-breaking priority;
- query individual genes or gene sets for family membership, thresholds, and activation counts;
- export CChains-ready `T_Network/data` and `N_Network/data` folders, including `sample.txt`, `names.csv`, and `nodes.txt`;
- run downstream stability analyses under synthetic augmentation or real-sample subsampling.

## Documentation

- `USER_MANUAL.md` describes the scientific workflow, inputs, outputs, command-line usage, gene queries, panel options, stability analyses, and validation cautions.
- `DEVELOPER_GUIDE.md` describes the internal Python structure, implementation contracts, testing scope, and extension rules.

## Authors

Gabriel Gil, Augusto González, and Julio César Drake.

## Reference

If you use GenePan in scientific work, please cite:

G. Gil, C. Carricarte, J. C. Drake-Pérez, Y. Perera, A. Gonzalez. Highly specific and sensitive gene panels for cancer screening: First application of only-normal and only-tumor genes. Tumor Discovery 2025, 4(3), 58-69. https://doi.org/10.36922/TD025190035
