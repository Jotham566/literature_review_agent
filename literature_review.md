```mermaid
graph TD
  A[Research Planning] --> B[Data Collection]
  B --> C[Analysis]
  C --> D[Draft Generation]
  D --> E[Fact Checking]
  E --> F[Citation Verification]
  F --> G[Quality Control]
  G --> H[Final Review]
  H --> |If issues found| D
  H --> |Approved| I[Output]
  ```