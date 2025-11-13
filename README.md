From Text to Network with LLMs - Module Assignment
Authors: Alexander Christiansen, Anders Skjødt Sønderby, Christian Ory Nielsen & Peter Christian Østerballe
Date: November 14, 2025
Dataset
Source: S&P 500 Earnings Call Transcripts (2005-2025)
Subset: Information Technology sector, 2024 only (74 companies)
Rationale:

Sector focus: IT companies share analysts with overlapping expertise, creating interpretable network patterns and distinct subsegments (chips, software, hardware, cloud)
Temporal scope: Single year (2024) provides recent market dynamics while keeping computational requirements manageable
Size: 74 companies offer sufficient relational data without overwhelming processing needs

Methodology
LLM Extraction Pipeline

Sector Classification: Used Ollama Gemma3:12b with few-shot learning to classify S&P 500 companies into GICS-like sectors, identifying 74 IT companies
Structured Extraction: Applied Google Gemini 2.5 Flash Lite with Pydantic schema validation to extract participant data (name, role, organization, type) from earnings call transcripts
Output: 4,000+ participant entries with ~2,500 unique speakers and ~200 analyst organizations

Network Construction

Network type: Undirected, weighted company-company network
Nodes: 74 IT companies (node size = unique analyst count)
Edges: Shared analyst relationships (edge weight = number of shared analysts)
Logic: Companies sharing analysts likely operate in similar market segments or are perceived as competitive peers

Network Analysis

Community detection: Greedy modularity optimization (modularity = 0.570)
Communities identified: 4 major groups (Semiconductors, Software/Cloud, Hardware, Mixed/FinTech)
Centrality metrics: Degree, eigenvector, and betweenness centrality calculated
Validation: Manual review of 20 random samples showed 90-95% precision and 85-90% recall

Main Findings

Distinct market segmentation: Modularity score (0.570) confirms clear IT subsectors aligned with industry intuition
Cybersecurity dominance: CrowdStrike, Palo Alto Networks, and ServiceNow show highest analyst coverage (30+ organizations each)
Semiconductor integration: Chip companies (NVIDIA, Broadcom, Qualcomm) form densest community, indicating high analyst specialization
Bridge companies: Western Digital, Broadcom, and Intel connect multiple communities (high betweenness centrality)
Distributed importance: No single dominant player (max eigenvector centrality = 0.26), reflecting diverse IT landscape

Limitations

Topic extraction quality: Topics too granular for pattern analysis; future work should use predefined taxonomy
Single time period: Cannot observe coverage evolution or temporal dynamics
Incomplete recall: Minor participants occasionally missed (~10-15%)
Large-cap bias: Only S&P 500 companies; findings may not generalize to small/mid-cap firms
Causality ambiguity: Shared coverage correlates with but doesn't prove strategic similarity
Network scope: Only explored company-company networks; alternatives (bipartite, multi-layer) not tested

Repository Structure
├── notebooks/
│   ├── LLM_pipeline_identifying_branches_cleaned.ipynb  # Sector classification
│   └── exam_load_3__1_.ipynb                            # Main analysis pipeline
├── data/
│   ├── it_companies.csv                                  # 74 IT companies list
│   ├── structured_output.csv                             # LLM-extracted data
│   └── manual_review_sample.csv                          # Quality evaluation
├── outputs/
│   └── Slide_deck.pptx                                   # Final presentation
└── requirements.txt                                      # Dependencies
Requirements
See requirements.txt for full dependencies. Key packages:

pandas, numpy, networkx (analysis)
matplotlib, seaborn (visualization)
openai, pydantic (LLM integration)
datasets (HuggingFace data loading)


This version is approximately 60% shorter than the original while maintaining all required elements. It focuses on the essential information and removes redundant details, examples, and extensive explanations.RetryAplease use this text with the same context you already have, but make this readme text much shorter and more concise:From Text to Network: Earnings Call Analysis with LLMs
Project Overview
This project demonstrates a complete pipeline for extracting structured information from unstructured text using Large Language Models (LLMs) and transforming it into network representations for analysis. We analyze earnings call transcripts from S&P 500 Information Technology companies to understand analyst coverage patterns and identify company communities based on shared analyst attention.Course: From Text to Network with LLMs - Module Assignment
Dataset: S&P 500 Earnings Call Transcripts (Information Technology Sector, 2024)
Analysis Focus: Analyst coverage networks revealing market segments and company relationshipsRepository Structure
.
├── README.md                                          # This file
├── requirements.txt                                   # Python dependencies
│
├── notebooks/
│   ├── LLM_pipeline_identifying_branches_cleaned.ipynb  # Initial sector classification
│   └── exam_load_3__1_.ipynb                            # Main analysis pipeline
│
├── data/
│   ├── it_companies.csv                               # List of 74 IT companies
│   ├── structured_output.csv                          # LLM-extracted participant data
│   └── manual_review_sample.csv                       # Sample for precision/recall evaluation
│
├── outputs/
│   └── Slide_deck.pptx                                # Final presentation (max 10 slides)
│
└── docs/
    └── Tekst_eksempler__precision-recall_.docx        # Manual quality assessment notes
Dataset Selection & Rationale
Dataset: S&P 500 Earnings Call Transcripts
Source: kurry/sp500_earnings_transcripts
Full dataset: Earnings call transcripts for S&P 500 and US large caps, 2005–2025
Our Subset Choice: Information Technology Sector Only (2024)
Why Information Technology?Coherent analytical scope: Focusing on a single sector ensures meaningful comparisons and interpretable network patterns
Rich relational potential: IT companies share analysts with overlapping expertise, creating dense network structures
Manageable scope: 74 companies provide sufficient data without overwhelming processing requirements
Business relevance: IT sector represents distinct subsegments (chips, software, hardware, cloud) ideal for community detection
Why 2024 only?Recent data ensures current market dynamics
Keeps computational requirements reasonable
Single time period simplifies analysis while maintaining richness
LLM Extraction Process
1. Sector Classification (Preprocessing)
Notebook: LLM_pipeline_identifying_branches_cleaned.ipynbTool: Ollama with Gemma3:12b model
Task: Classify all S&P 500 companies into 11 GICS-like sectors
Method: Few-shot classification with structured JSON output
Result: Identified 74 Information Technology companies for subsequent analysisKey Features:Batch processing (80 companies per batch) for efficiency
Caching mechanism to avoid redundant API calls
Confidence scoring for classification quality assessment
Few-shot examples improve accuracy (Apple→IT, Pfizer→Health Care, etc.)
2. Structured Participant Extraction
Notebook: exam_load_3__1_.ipynbTool: Google Gemini 2.5 Flash Lite via OpenAI-compatible API
Task: Extract structured information about all speakers in earnings call transcriptsExtraction Schema (Pydantic)
class Participant(BaseModel):
    person_name: str           # Speaker's full name
    role: str                  # Job title (CEO, CFO, Analyst, etc.)
    organization: str          # Company/firm they represent
    type: str                  # 'company_rep' or 'analyst'
    topics: List[str]          # Discussion topics (not used in network)class EarningsCallStructure(BaseModel):
    company_name: str
    participants: List[Participant]
Why This Schema?
Person-centric: Earnings calls are conversations where roles matter
Clear classification: Binary distinction (company representative vs. external analyst)
Relational potential: Enables analyst-company and company-company networks
Structured output: Pydantic + JSON Schema enforcement ensures consistent data format
Processing Pipeline
API Configuration: OpenAI client pointed to Gemini endpoint for fast, cheap inference
Schema Validation: JSON Schema enforcement via response_format parameter
Batch Processing: Process all 2024 IT sector transcripts with progress tracking
Data Flattening: Convert nested structures to flat DataFrame (one row per participant per call)
Error Handling: Graceful failure management, skip problematic transcripts
Output: structured_output.csv with 4,000+ participant entries across 74 companiesDescriptive Exploration & Quality Assessment
Dataset Statistics
Total Records: 4,000+ participant entries from IT sector earnings calls (2024)
Unique Companies: 74 Information Technology companies from S&P 500
Unique Participants: ~2,500 individual speakers
Unique Organizations: ~200 analyst firms and company organizations
Participant Distribution
Analysts: ~70% of participants (investment analysts from financial institutions)
Company Representatives: ~30% (executives, officers, IR teams)
Top Analyst Organizations
Morgan Stanley
Goldman Sachs
JPMorgan
Bank of America
Evercore
Analyst Coverage Patterns
Coverage Range: High-coverage companies tracked by 30+ analyst organizations; low-coverage by <10
Average: ~15-20 analyst organizations per company
Top 3 Most Covered:
CrowdStrike Holdings - 34 unique analyst organizations
ServiceNow, Inc. - 33 unique analyst organizations
Palo Alto Networks, Inc. - 32 unique analyst organizations
Quality Assessment: Precision & Recall Analysis
Method
File: manual_review_sample.csv
Sample Size: 20 randomly selected participant entries
Comparison: Manual review against original transcriptsFindings
✅ Strengths (High Precision)
Type Classification (analyst vs. company_rep): Nearly perfect accuracyModel correctly identifies participant roles based on context
Very few misclassifications
Organization Extraction: High accuracyAnalyst firms correctly identified (e.g., "Morgan Stanley", "Goldman Sachs")
Company affiliations properly captured
Information is typically explicit in transcripts ("Ben Reitzes from Melius Research")
Name and Role Extraction: Generally accurateSpeaker names captured correctly
Job titles appropriately extracted (CEO, CFO, Analyst, etc.)
⚠️ Limitations (Lower Recall/Issues)
Topic Extraction Quality:Problem: Topics are highly granular and often unique per participant
Impact: Difficult to identify patterns or aggregate insights
Root Cause: No predefined topic taxonomy in prompt
Solution for Future Work: Provide LLM with 5-10 predetermined topic categories
Completeness:Some participants may be missed if mentioned only briefly
Minor speakers or passing mentions sometimes not extracted
Topic Inconsistency:Same topic described differently across extractions
Example: "AI strategy" vs "artificial intelligence initiatives" vs "AI adoption"
Overall Assessment
Precision: High (~90-95%) - What the model extracts is generally correct
Recall: Good (~85-90%) - Model captures most participants, occasional misses
Usability: Excellent for network construction (participants & organizations are reliable)Impact on Analysis
Decided NOT to use topics in network analysis due to quality issues
Focus on participant-company relationships where extraction quality is high
This demonstrates good scientific practice: acknowledge and work around limitations
Network Construction: Knowledge Graph
Graph Definition
Type: Undirected, weighted, one-mode networkNodes (74 companies)
Entity: IT companies hosting earnings calls
Attribute: Number of unique analysts covering the company
Example: "NVIDIA Corporation" (node_size = 28 analysts)
Edges (Shared analyst coverage)
Relationship: Companies share one or more individual analysts
Weight: Number of analysts both companies share
Example: CrowdStrike ↔ Palo Alto Networks (weight = 17 shared analysts)
Rationale
This network represents shared analyst attention across IT companies:Market Segmentation: Companies sharing many analysts likely operate in similar market segments
Competitive Positioning: Shared coverage suggests companies are perceived as peers or competitors
Investor Perspective: Network reveals how the analyst/investor community groups companies
Construction Pipeline
Filter Data: Extract only analyst participants (exclude company representatives)
Create Nodes: One node per company with analyst count attribute
Calculate Edges: For each company pair, count shared individual analysts
Apply Threshold: Include edge only if ≥1 shared analyst (adjustable parameter)
Store Graph: NetworkX Graph object with full metadata
Implementation: create_company_similarity_network() function in main notebookNetwork Analysis Results
1. Community Detection
Method: Greedy Modularity Optimization
Algorithm: Iteratively merge nodes to maximize modularity score
Constraint: Communities must have ≥3 members
Visualization: Top 4 largest communities displayedModularity Score: 0.570
Interpretation:Moderate-to-strong community structure (>0.3 is considered good)
More connections within communities than expected by chance
Indicates meaningful market segmentation in the IT sector
Not perfect (would be 1.0), but significantly better than random
Identified Communities
Community 1 (Chip/Semiconductor) - Light GreenCompanies: NVIDIA, Broadcom, Microchip Technology, Qualcomm, Intel, AMD, etc.
Characteristic: Dense interconnectedness, strong shared analyst coverage
Insight: Analysts specializing in semiconductor market cover these companies as a group
Community 2 (Software/Cloud) - Dark BlueCompanies: Microsoft, Adobe, CrowdStrike, ServiceNow, Salesforce, Oracle
Characteristic: Moderately connected, strong edges between major pairs (Adobe-Microsoft)
Insight: Enterprise software and cloud infrastructure companies
Community 3 (Hardware/Consumer Electronics) - RedCompanies: Apple, Dell, HP Inc., Hewlett Packard Enterprise
Characteristic: Consumer electronics and enterprise hardware manufacturers
Insight: Companies selling physical computing devices
Community 4 (Mixed/Financial IT) - OrangeCompanies: Smaller, more diverse group including FinTech and specialized IT services
Characteristic: Less dense, varied analyst coverage patterns
2. Centrality Analysis
Most Covered Companies (By Analyst Count)
CrowdStrike - 34 analysts
ServiceNow - 33 analysts
Palo Alto Networks - 32 analysts
Salesforce - 31 analysts
Adobe - 30 analysts
Insight: High-growth software/cybersecurity companies attract most analyst attentionMost Connected Companies (By Weighted Degree)
Broadcom - 171 total shared analyst connections
Qualcomm - 158 connections
Western Digital - 155 connections
Intel - 153 connections
NVIDIA - 151 connections
Insight: Central companies with broad connections across IT sectorEigenvector Centrality (Top 5)
Western Digital - 0.26
Broadcom - 0.25
Seagate Technology - 0.24
Qualcomm - 0.24
Microchip Technology - 0.23
Interpretation:Moderate scores (0.26 is not exceptionally high)
No single "super-central" company dominates analyst attention
Suggests multiple distinct IT subsectors with distributed importance
These companies bridge between different communities
Business Meaning: Western Digital and Broadcom connect multiple IT segments, indicating versatile market positioning that appeals to diverse analyst specializationsBetweenness Centrality (Top 5)
Dayforce Inc. - 0.21
Autodesk - 0.19
Western Digital - 0.18
Intel - 0.17
Seagate Technology - 0.16
Interpretation:These companies act as "bridges" between different communities
Dayforce connects HR tech analysts with broader IT sector coverage
Intel bridges semiconductor and hardware communities
High betweenness = strategic position connecting distinct market segments
3. Strongest Company Pairs (Most Similar)
CrowdStrike ↔ Palo Alto Networks - 17 shared analystsBoth cybersecurity leaders, direct competitors
ServiceNow ↔ Salesforce - 16 shared analystsEnterprise cloud platforms, similar market positioning
Adobe ↔ Microsoft - 15 shared analystsEnterprise software giants with overlapping product portfolios
Broadcom ↔ Qualcomm - 15 shared analystsMajor semiconductor designers, mobile/wireless focus
NVIDIA ↔ Broadcom - 14 shared analystsLeading chip designers, AI/datacenter market leaders
Main Findings
Key Insights
Market Segmentation is RealModularity score (0.570) confirms distinct IT subsectors exist
Analyst coverage patterns align with industry intuition (chips, software, hardware)
Communities represent genuine market segments, not random clustering
Cybersecurity is HotCrowdStrike, Palo Alto Networks, and CrowdStrike top analyst coverage
Strongest company pair relationship (17 shared analysts)
Indicates high investor/analyst interest in security companies
Central Players Bridge CommunitiesCompanies like Intel, Western Digital bridge multiple segments
High betweenness centrality indicates strategic market positioning
These companies relevant to analysts across specializations
Chip Sector is Tightly IntegratedSemiconductor community (NVIDIA, Broadcom, Qualcomm, etc.) shows densest interconnections
Analysts covering one chip company likely cover many others
Reflects high specialization in semiconductor analysis
No Single Dominant PlayerModerate eigenvector centrality scores (max 0.26)
Multiple important companies rather than one central giant
Reflects diverse, multi-faceted IT sector landscape
Business Implications
For Companies: Understand competitive positioning through analyst coverage
For Investors: Identify market segments and peer groups for comparative analysis
For Analysts: See how coverage patterns reveal market structure
For Researchers: Validate that LLM-extracted networks match domain knowledge
Limitations & Future Work
1. LLM Extraction Limitations
Topic Extraction Quality
Issue: Topics are too granular and unique per participant
Impact: Cannot aggregate or find patterns in discussion topics
Solution: Provide LLM with predefined topic taxonomy (e.g., 5-10 categories like "AI/ML", "Cloud", "Cybersecurity", "Supply Chain", "Competition")
Completeness
Issue: Minor participants occasionally missed
Impact: Some analysts may not be captured if mentioned briefly
Mitigation: Focus on major participants reduces impact
2. Temporal Limitations
Single Time Period (2024 only)
Issue: Cannot observe evolution over time
Missing Insights: How analyst coverage shifts after events (earnings beats, product launches, acquisitions)
Future Work: Multi-year analysis to track network dynamics
Individual Analyst vs. Firm Level
Current: Network based on individual analysts (person-level)
Alternative: Could aggregate to firm level (e.g., "Morgan Stanley" as single node)
Tradeoff: Individual level = more granular but may miss institutional patterns
3. Network Construction Choices
Shared Analyst Metric
Assumption: More shared analysts = more similar companies
Limitation: Doesn't account for why analysts cover both (genuine similarity vs. portfolio diversification)
Alternative Metrics: Could weight by analyst firm size, coverage intensity, or sentiment
Single Network Type
Current: Company-company network via shared analysts
Alternatives not explored:
Bipartite network (companies + analysts)
Topic-based networks (if topics were cleaner)
Temporal networks (across quarters)
4. Data Scope Limitations
IT Sector Only
Benefit: Coherent analysis
Cost: Cannot compare cross-sector patterns (e.g., IT vs. Healthcare analyst behavior)
Future Work: Multi-sector analysis to find universal vs. sector-specific patterns
S&P 500 Only
Bias: Only large-cap companies
Missing: Small/mid-cap IT companies with different analyst coverage
Consideration: Results may not generalize beyond large-cap universe
5. Methodological Considerations
Precision vs. Recall Trade-off
Precision: High for participant identification (~90-95%)
Recall: Good but not perfect (~85-90%)
Impact: Network likely captures major patterns but may miss edge cases
Community Detection Algorithm Choice
Method Used: Greedy modularity
Alternatives: Louvain, Label Propagation, Spectral Clustering
Consideration: Different algorithms may reveal different community structures
6. Interpretation Limitations
Causality
Network shows: Shared analyst attention
Network doesn't show: Why analysts cover certain combinations
Caution: Correlation ≠ causation (shared coverage ≠ strategic similarity)
Analyst Motivations
Assumption: Analyst coverage reflects perceived company similarity
Reality: Coverage also driven by client demand, firm strategy, analyst expertise
Complexity: Multiple factors influence analyst assignment
Future Enhancements
Improved Topic ExtractionUse predefined topic taxonomy in LLM prompt
Implement topic clustering post-processing
Explore topic modeling (LDA, BERTopic) as complement
Temporal AnalysisExpand to multiple years (2020-2024)
Track community evolution over time
Identify events that shift analyst coverage patterns
Multi-Level NetworksAnalyst firm level aggregation
Bipartite networks (companies ↔ analysts)
Multi-layer networks (analysts + topics + companies)
Alternative Similarity MetricsWeight by analyst seniority or firm size
Incorporate temporal patterns (coverage duration)
Use sentiment from transcript content
Cross-Sector ComparisonReplicate analysis across all 11 GICS sectors
Compare community structures
Identify sector-specific vs. universal patterns
Validation StudiesCompare with traditional industry classifications (GICS, NAICS)
Validate against actual company financials (revenue correlations)
Survey analysts about their coverage logic
Technical Requirements
Python Dependencies
# Core Data & Analysis
pandas>=2.0.0
numpy>=1.24.0
networkx>=3.0# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0# LLM & API
openai>=1.0.0
pydantic>=2.0.0
python-dotenv>=1.0.0# Data Loading
datasets>=2.14.0  # Hugging Face datasets
tqdm>=4.65.0      # Progress bars# Optional (for preprocessing)
ollama>=0.1.0     # Local LLM inference
Contributions
Group Members: Alexander Christiansen, Anders Skjødt Sønderby, Christian Ory Nielsen & Peter Christian Østerballe
Module assignment From Text to Network with LLMs
Submission Date: November 14, 2025, 10:00 AMShow more2 / 2From Text to Network with LLMs
Authors: Alexander Christiansen, Anders Skjødt Sønderby, Christian Ory Nielsen & Peter Christian Østerballe
Course: From Text to Network with LLMs - Module Assignment
Date: November 14, 2025
Dataset
Source: S&P 500 Earnings Call Transcripts (2005-2025)
Subset: Information Technology sector, 2024 (74 companies, 4,000+ participant entries)
Rationale:

Sector focus: IT companies share analysts, enabling analysis of market segmentation (chips, software, hardware, cloud)
Temporal scope: Single year provides current dynamics with manageable computational requirements
Relational richness: Sufficient overlap for meaningful network construction

Methodology
LLM Extraction

Sector classification: Ollama Gemma3:12b with few-shot learning identified 74 IT companies from S&P 500
Participant extraction: Google Gemini 2.5 Flash Lite with Pydantic schema extracted structured data (name, role, organization, type) from transcripts
Quality: Manual review showed 90-95% precision, 85-90% recall for participant identification

Network Construction

Type: Undirected, weighted company-company network
Nodes: 74 IT companies (size = unique analyst count)
Edges: Shared analyst relationships (weight = number of shared analysts)
Logic: Companies sharing analysts likely operate in similar segments or are competitive peers

Analysis

Community detection: Greedy modularity optimization (modularity = 0.570)
Centrality metrics: Degree, eigenvector, and betweenness centrality
Result: 4 communities identified (Semiconductors, Software/Cloud, Hardware, Mixed/FinTech)

Main Findings

Distinct market segmentation: Modularity (0.570) confirms clear IT subsectors aligned with industry structure
Cybersecurity dominance: CrowdStrike, Palo Alto, ServiceNow show highest coverage (30+ analyst organizations)
Semiconductor integration: Chip companies form densest community, indicating high analyst specialization
Bridge companies: Western Digital, Broadcom, Intel connect multiple segments (high betweenness centrality)
Distributed landscape: No single dominant player (max eigenvector = 0.26), reflecting diverse IT sector

Limitations

Topic extraction: Too granular for pattern analysis; future work needs predefined taxonomy
Temporal scope: Single period limits analysis of coverage evolution
Incomplete recall: ~10-15% of minor participants missed
Large-cap bias: S&P 500 only; may not generalize to smaller firms
Causality: Shared coverage correlates with but doesn't prove strategic similarity
Network scope: Only explored company-company networks; bipartite and temporal alternatives not tested
Algorithm choice: Greedy modularity used; other methods (Louvain, spectral) may reveal different structures

Repository Structure
├── notebooks/
│   ├── LLM_pipeline_identifying_branches_cleaned.ipynb  # Sector classification
│   └── exam_load_3__1_.ipynb                            # Main pipeline
├── data/
│   ├── it_companies.csv                                  # 74 companies
│   ├── structured_output.csv                             # Extracted data
│   └── manual_review_sample.csv                          # Quality eval
├── outputs/
│   └── Slide_deck.pptx                                   # Presentation
└── requirements.txt                                      # Dependencies
Requirements
See requirements.txt. Key packages: pandas, numpy, networkx, matplotlib, seaborn, openai, pydantic, datasets
