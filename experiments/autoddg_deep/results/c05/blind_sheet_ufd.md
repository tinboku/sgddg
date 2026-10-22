# C05 blind evaluation sheet (UFD)

Fill in without opening `blind_key.json`.

## 10811fc1-8ad0-4d8b-bdcd-cde9b43c86e6

### Profile (original form)

```
The key data profile information for this dataset includes:
**rep_file_num**: Data is of type text. There are 925 unique values. 
**CIK**: Data is of type integer. There are 926 unique values. Coverage spans from 40417.0 to 1729678.0. 
**entity_name**: Data is of type text. 
**street1**: Data is of type text. There are 346 unique values. 
**street2**: Data is of type text. There are 177 unique values. 
**city**: Data is of type text. There are 97 unique values. 
**state_code**: Data is of type text. There are 33 unique values. 
**zip**: Data is of type text. There are 185 unique values. 
**filing_date**: Data is of type text. There are 214 unique values. 
**doc_type_code**: Data is of type text. There are 72 unique values. 
**Temporal coverage** for columns filing_date with resolution day, covering from 2016-11-02 to 2018-07-18.
- Temporal scope: filing_date: 1994-08-05 to 2018-07-18 (of 100% parseable values)
- Geographic scope: city: probably city names (not confirmed); zip (zip code) includes 60606, 19809, 02110, 10019, 10022 and others (185 distinct)
- Measures and units: CIK: 2230 to 1745846 (median 1380187)
- Representative categories: rep_file_num: 811-06660 (0%), 811-00041, 811-22603, 811-22573, 811-22575 (925 distinct values); street1: 100 BELLEVUE PARKWAY (8%), 333 WEST WACKER DRIVE, TWO INTERNATIONAL PLACE, 1633 BROADWAY, 620 EIGHTH AVENUE (346 distinct values); street2: [NULL] (55%), 49TH FLOOR, SUITE 1800, MUTUAL FUND DEPARTMENT, SUITE 700 (177 distinct values); state_code: NY (30%), IL, MA, DE, PA (33 distinct values)
```

### Description A

This dataset focuses on Closed-End Funds and includes detailed information about various financial entities. It contains 925 unique identification numbers (rep_file_num) and 926 unique Central Index Key (CIK) values, covering a temporal range from August 5, 1994, to July 18, 2018. The dataset features spatial data such as street addresses, cities, state codes, and zip codes, with 346 unique street addresses and 185 distinct zip codes. The filing dates provide insights into the documentation process for these funds. This dataset can be utilized for analyzing trends in closed-end fund filings and understanding the geographical distribution of these financial entities.

### Description B

This dataset focuses on Closed-End Funds, providing detailed information about various funds, including their names, addresses, filing dates, and document types. It contains 925 unique entries, with data spanning from 1994 to 2018. The dataset includes key identifiers such as the CIK (Central Index Key) and various location details like street addresses, cities, and zip codes. Each column has a specific semantic profile, representing organizational and location data relevant to the financial domain. This dataset can be utilized for analyzing trends in closed-end funds, understanding their geographical distribution, and examining filing patterns over time.

| criterion | A | B |
|---|---|---|
| column confusion (value attributed to the wrong column), count | | |
| statements not supported by the profile, count | | |
| key facts missing (coverage, units, measures) | | |
| readability / usefulness 1-5 | | |
| preferred (A/B/tie) | | |

## 1ff67fc2-ff44-48ea-8b9b-d19bc71c9e08

### Profile (original form)

```
The key data profile information for this dataset includes:
**ADDRESS**: Data is of type text. 
**BKCLASS**: Data is of type text. There are 7 unique values. 
**CBSA**: Data is of type text. There are 933 unique values. 
**CBSA_DIV**: Data is of type text. There are 31 unique values. 
**CBSA_DIV_FLG**: Data is of type integer. There are 2 unique values. Coverage spans from 0.0 to 1.0. 
**CBSA_DIV_NO**: Data is of type integer. There are 32 unique values. Coverage spans from 0.0 to 48864.0. 
**CBSA_METRO**: Data is of type integer. There are 389 unique values. Coverage spans from 0.0 to 47900.0. 
**CBSA_METRO_FLG**: Data is of type integer. There are 2 unique values. Coverage spans from 0.0 to 1.0. 
**CBSA_METRO_NAME**: Data is of type text. There are 388 unique values. 
**CBSA_MICRO_FLG**: Data is of type integer. There are 2 unique values. Coverage spans from 0.0 to 1.0. 
**CBSA_NO**: Data is of type integer. There are 934 unique values. Coverage spans from 0.0 to 48620.0. 
**CERT**: Data is of type integer. There are 4379 unique values. Coverage spans from 628.0 to 58534.0. 
**CITY**: Data is of type text. There are 10271 unique values. 
**COUNTY**: Data is of type text. There are 1918 unique values. 
**CSA**: Data is of type text. There are 174 unique values. 
**CSA_FLG**: Data is of type integer. There are 2 unique values. Coverage spans from 0.0 to 1.0. 
**CSA_NO**: Data is of type integer. There are 175 unique values. Coverage spans from 0.0 to 548.0. 
**ESTYMD**: Data is of type text. There are 21387 unique values. 
**FI_UNINUM**: Data is of type integer. There are 4379 unique values. Coverage spans from 417.0 to 76173.0. 
**MAINOFF**: Data is of type integer. There are 2 unique values. Coverage spans from 0.0 to 0.0. 
**NAME**: Data is of type text. 
**OFFNAME**: Data is of type text. There are 47324 unique values. 
**OFFNUM**: Data is of type integer. There are 9739 unique values. Coverage spans from 0.0 to 5635.0. 
**RUNDATE**: Data is of type text. There are 1 unique values. 
**SERVTYPE**: Data is of type integer. There are 15 unique values. Coverage spans from 11.0 to 11.0. 
**STALP**: Data is of type text. There are 58 unique values. 
**STCNTY**: Data is of type integer. There are 3204 unique values. Coverage spans from 1021.0 to 55059.0. 
**STNAME**: Data is of type text. There are 58 unique values. 
**UNINUM**: Data is of type integer. There are 78300 unique values. Coverage spans from 1087.0 to 665545.0. 
**ZIP**: Data is of type integer. There are 18108 unique values. Coverage spans from 801.0 to 97720.0. 
**Temporal coverage** for columns ESTYMD with resolution day, covering from 1869-01-01 to 2023-10-16.
**Temporal coverage** for columns RUNDATE with resolution day, covering from 2026-07-17 to 2026-07-17.
- Temporal scope: ESTYMD: 1800-01-01 to 2025-10-16 (excluding 5 values outside 1800-2100 (e.g. 01/01/1792, 05/29/1792)); RUNDATE: 2026-07-17 to 2026-07-17
- Geographic scope: CITY: probably city names (not confirmed); COUNTY: probably county names (not confirmed); ZIP (zip code) includes 00000, 10022, 33134, 75225, 75034 and others (18108 distinct)
- Measures and units: CBSA_DIV_FLG: 0 to 1 (median 0); CBSA_METRO: 0 to 49740 (median 25420); CBSA_METRO_FLG: 0 to 1 (median 1); CBSA_MICRO_FLG: 0 to 1 (median 0)
- Representative categories: ADDRESS: 101 N Main St (0%), 101 S Main St, 101 E Main St, 201 N Main St, 101 W Main St (73220 distinct values); BKCLASS: N (46%), NM, SM, SI, SB (7 distinct values)
```

### Description A

The Bank Locations Dataset provides detailed information about various bank branches across the United States, including their addresses, city and county affiliations, and service types. It covers a temporal range from January 1, 1800, to October 16, 2025, with a specific run date of July 17, 2026. The dataset includes spatial data such as city names, county names, and ZIP codes, making it useful for geographic analysis and location-based services. This dataset can be utilized for market research, branch location planning, and understanding banking accessibility in different regions. It contains a rich variety of unique identifiers and classifications for effective data analysis.

### Description B

The Bank Locations Dataset provides detailed information on bank branches across various locations in the United States, with data covering 43,000 entries from 1869 to 2023. Each entry includes essential details such as the address, city, county, and ZIP code, allowing for effective geolocation and analysis of banking services. The dataset features a rich semantic profile, representing locations at various levels, including street addresses, cities, and states. This dataset can be utilized for market analysis, identifying banking trends, and enhancing financial service accessibility in different regions. It serves as a valuable resource for researchers, financial institutions, and policymakers.

| criterion | A | B |
|---|---|---|
| column confusion (value attributed to the wrong column), count | | |
| statements not supported by the profile, count | | |
| key facts missing (coverage, units, measures) | | |
| readability / usefulness 1-5 | | |
| preferred (A/B/tie) | | |

## 3e23114a-2657-4880-8add-f3c24425f937

### Profile (original form)

```
The key data profile information for this dataset includes:
**Year**: Data is of type text. There are 17 unique values. 
**Sex**: Data is of type text. There are 3 unique values. 
**Age**: Data is of type text. There are 9 unique values. 
**Race and Hispanic Origin**: Data is of type text. There are 4 unique values. 
**State**: Data is of type text. There are 52 unique values. 
**Deaths**: Data is of type integer. There are 1522 unique values. Coverage spans from 15.0 to 3220.0. 
**Population**: Data is of type integer. There are 2703 unique values. Coverage spans from 648414.0 to 101042357.0. 
**Crude Death Rate**: Data is of type float. Coverage spans from 0.1 to 36.9. 
**Standard Error for Crude Rate**: Data is of type float. Coverage spans from 0.0 to 1.6. 
**Low Confidence Limit for Crude Rate**: Data is of type float. Coverage spans from 0.1 to 35.8. 
**Upper Confidence Limit for Crude Rate**: Data is of type float. Coverage spans from 0.2 to 37.9. 
**Age-adjusted Rate**: Data is of type float. Coverage spans from 2.9 to 29.9. 
**Standard Error Age-adjusted Rate**: Data is of type float. Coverage spans from 0.1 to 1.7. 
**Lower Confidence Limit for Age-adjusted rate**: Data is of type float. Coverage spans from 2.1 to 28.2. 
**Upper Confidence Limit for Age-adjusted Rate**: Data is of type float. Coverage spans from 3.7 to 31.4. 
**State Crude Rate in Range**: Data is of type text. There are 60 unique values. 
**US Crude Rate**: Data is of type float. Coverage spans from 6.0 to 16.3. 
**US Age-adjusted Rate**: Data is of type float. Coverage spans from 6.1 to 16.3. 
**Temporal coverage** for columns Year with resolution year, covering from 1999-01-01 to 2015-01-01.
- Identity: 2703 rows, 18 columns
- Temporal scope: Year: 1999 to 2015; Year: consecutive time points are one year apart
- Measures and units: Deaths: 1 to 52404 (median 478); Population: 238163 to 321418820 (median 6403290); Crude Death Rate: 0 to 53.5 (median 8.8); Standard Error for Crude Rate: 0 to 1.8 (median 0.4)
- Representative categories: Sex: Both Sexes (55%), Male, Female; Age: All Ages (40%), 15-24 years, Less than 15 years, 35-44 years, 45-54 years (9 distinct values); Race and Hispanic Origin: All Races-All Origins (49%), Hispanic, Non-Hispanic White, Non-Hispanic Black; State: United States (68%), Texas, New Hampshire, Nevada, North Carolina (52 distinct values)
- Limitations: Age-adjusted Rate: 60% missing
```

### Description A

This dataset focuses on Drug Poisoning Mortality, providing insights into death rates associated with drug overdoses across various demographics in the United States from 1999 to 2015. It includes 2,703 rows and 18 columns, capturing data on factors such as year, sex, age, race and Hispanic origin, state, deaths, and population. The semantic profile of the dataset highlights its representation of demographic and healthcare measurements, allowing for comprehensive statistical analysis. Researchers and policymakers can utilize this dataset to identify trends, assess risk factors, and develop targeted interventions to address drug poisoning mortality effectively.

### Description B

This dataset focuses on drug poisoning mortality in the United States, covering the years 1999 to 2015. It includes 2,703 rows and 18 columns, detailing deaths, population figures, and various demographic factors such as sex, age, and race. The dataset provides valuable insights into mortality rates, including crude and age-adjusted death rates, which can be used for public health analysis and policy-making. It captures data across 52 states, allowing for geographic comparisons. However, it is important to note that approximately 60% of the age-adjusted rate data is missing, which may affect certain analyses.

| criterion | A | B |
|---|---|---|
| column confusion (value attributed to the wrong column), count | | |
| statements not supported by the profile, count | | |
| key facts missing (coverage, units, measures) | | |
| readability / usefulness 1-5 | | |
| preferred (A/B/tie) | | |

## 4b5bd984-71d8-4d76-b8f5-949532b5b5a7

### Profile (original form)

```
The key data profile information for this dataset includes:
**Service Request #**: Data is of type text. There are 371479 unique values. 
**Request Type**: Data is of type text. There are 40 unique values. 
**Request Reason**: Data is of type text. There are 184 unique values. 
**Date Created**: Data is of type text. There are 289854 unique values. 
**Date Modified**: Data is of type text. There are 206617 unique values. 
**Date Closed**: Data is of type text. There are 110031 unique values. 
**Request Status**: Data is of type text. There are 2 unique values. 
**Responsible Agency**: Data is of type text. There are 28 unique values. 
**Address**: Data is of type text. There are 167367 unique values. 
**Council District**: Data is of type text. There are 5 unique values. 
**Status**: Data is of type text. There are 49 unique values. 
**Contractor**: Data is of type text. There are 9 unique values. 
**Contractor Action**: Data is of type text. There are 11 unique values. 
**RowID**: Data is of type integer. There are 371479 unique values. Coverage spans from 485769.0 to 980914.0. 
**X**: Data is of type float. Coverage spans from 3663170.70315 to 3720107.92166. 
**Y**: Data is of type float. Coverage spans from 517262.945141 to 569711.089126. 
**Longitude**: Data is of type float. Coverage spans from -90.1253749662 to -89.9341022195143. 
**Latitude**: Data is of type float. Coverage spans from 29.9187750214 to 30.057648870345684. 
**Location**: Data is of type geocoordinates. 
**Temporal coverage** for columns Date Created with resolution second, covering from 2019-01-02 to 2023-03-06.
**Temporal coverage** for columns Date Modified with resolution second, covering from 2020-03-24 to 2026-10-15.
**Temporal coverage** for columns Date Closed with resolution second, covering from 2012-07-26 to 2026-10-12.
**Spatial coverage** for columns Latitude, Longitude, with type latlong.
**Spatial coverage** for columns Location, with type point_latlong.
- Temporal scope: Date Created: 2019-01-01 21:33:04 to 2026-10-20 23:46:58; Date Modified: 2019-01-03 16:28:02 to 2026-10-20 23:47:01; Date Closed: 2012-03-25 19:00:00 to 2026-10-20 13:09:13
- Geographic scope: coordinates within longitude -90.13 to -89.93, latitude 29.91 to 30.06; Council District: probably district names (not confirmed)
- Measures and units: RowID: 194451 to 1330219 (median 672783); X: 537829 to 3785502 (median 3680266); Y: 508696 to 609487 (median 535952); Longitude: -99.98 to 0 (median -90.07)
- Representative categories: Request Type: Trash/Recycling (46% of 369722 non-blank), Property Maintenance, Roads/Drainage, Abandoned Vehicles, Traffic Signals/Signs/Striping/Streetlights (40 distinct values)
```

### Description A

This dataset contains 311 service requests from New Orleans, covering various issues reported by residents, such as missed trash pick-ups, street light outages, and property maintenance. The data spans from January 2, 2019, to March 6, 2023, and includes details like request type, reason, status, and responsible agency. Each entry is identified by a unique service request number. The semantic profile highlights the dataset's categorization of incidents and organizations involved, making it useful for analyzing community needs and response effectiveness. This dataset can aid in urban planning, resource allocation, and improving city services.

### Description B

This dataset contains information on 311 service requests made between January 1, 2019, and October 20, 2026. It includes details such as the type and reason for each request, the responsible agency, and the status of the request. The dataset features 371,479 unique service request identifiers and covers various categories, with a significant focus on trash and recycling issues. The spatial coverage includes coordinates within the longitude range of -90.13 to -89.93 and latitude range of 29.91 to 30.06. This dataset can be utilized for analyzing service request trends, identifying areas needing attention, and improving municipal services.

| criterion | A | B |
|---|---|---|
| column confusion (value attributed to the wrong column), count | | |
| statements not supported by the profile, count | | |
| key facts missing (coverage, units, measures) | | |
| readability / usefulness 1-5 | | |
| preferred (A/B/tie) | | |

## 682835a9-48ce-44f3-a325-ceda77a53975

### Profile (original form)

```
The key data profile information for this dataset includes:
**Filing Type**: Data is of type text. There are 3 unique values. 
**Original Filing Date**: Data is of type text. There are 69 unique values. 
**Period Months**: Data is of type text. 
**Period Year**: Data is of type integer. There are 11 unique values. Coverage spans from 2018.0 to 2026.0. 
**Period Start Date**: Data is of type text. There are 52 unique values. 
**Period End Date**: Data is of type text. There are 53 unique values. 
**Permit Consultant Name**: Data is of type text. There are 113 unique values. 
**Permit Consultant Employer**: Data is of type text. 
**Permit Consultant Address**: Data is of type text. 
**Permit Consultant Telephone**: Data is of type text. There are 123 unique values. 
**Permit Consultant Email**: Data is of type text. There are 95 unique values. 
**Additional Clients Sheets Required**: Data is of type text. There are 1 unique values. 
**Additional Permit Sheets Required**: Data is of type text. There are 1 unique values. 
**Additional Contributions Sheets Required**: Data is of type text. There are 1 unique values. 
**Signer Name**: Data is of type text. There are 85 unique values. 
**Date Signed**: Data is of type text. There are 1349 unique values. 
**Filing**: Data is of type text. There are 1558 unique values. 
**DocuSign ID**: Data is of type text. There are 1248 unique values. 
**Temporal coverage** for columns Original Filing Date with resolution day, covering from 2018-01-10 to 2026-01-15.
**Temporal coverage** for columns Period Start Date with resolution quarter, covering from 2015-07-15 to 2026-01-01.
**Temporal coverage** for columns Period End Date with resolution month, covering from 2015-09-30 to 2026-03-31.
**Temporal coverage** for columns Date Signed with resolution minute, covering from 2015-04-15 to 2026-10-05.
- Identity: 1561 rows, 18 columns
- Temporal scope: Original Filing Date: 2015-10-15 to 2026-10-14 (no rows for 2016); Period Year: 1924 to 2026 (excluding 1 values outside 1800-2100 (e.g. 2202.0); most values 2018 to 2026; 1 far outside (e.g. 1924.0)); Period Year: consecutive time points are mostly one year apart (89% of steps); Period Start Date: consecutive time points are mostly one quarter apart (90% of steps); Period End Date: consecutive time points are mostly one quarter apart (90% of steps); Date Signed: 2015-03-23 00:00:00 to 2026-10-17 01:29:53
- Measures and units: Period Months unit: Months
- Representative categories: Filing Type: Original (89%), Amendment, Termination
- Limitations: Period Year: 20% missing
```

### Description A

This dataset contains 1,561 rows of Permit Consultant Reports, covering the period from 2015 to 2026. It includes various filing types such as Original, Amendment, and Termination, with a focus on the documentation related to permit consulting activities. The dataset features key information such as the names and contact details of permit consultants, their employers, and the dates of filing. Each entry is associated with a unique DocuSign ID for reference. The semantic profile of the dataset highlights the representation of temporal entities, organizations, and individuals, making it a valuable resource for analyzing trends in permit consulting and compliance reporting.

### Description B

This dataset contains 1,561 rows and 18 columns, focusing on Permit Consultant Reports. It includes various filing types such as Original, Amendment, and Termination, with temporal coverage spanning from October 15, 2015, to October 14, 2026. The dataset features detailed information about permit consultants, including their names, employers, contact details, and the dates associated with filings. The semantic profile highlights the dataset's utility for tracking permit-related activities and analyzing trends over time. This dataset can be valuable for regulatory compliance, research, and understanding the dynamics of permit consulting in urban planning.

| criterion | A | B |
|---|---|---|
| column confusion (value attributed to the wrong column), count | | |
| statements not supported by the profile, count | | |
| key facts missing (coverage, units, measures) | | |
| readability / usefulness 1-5 | | |
| preferred (A/B/tie) | | |

## be2a86f4-03f6-4e89-90b3-cc122c247b74

### Profile (original form)

```
The key data profile information for this dataset includes:
**index**: Data is of type text. There are 2227 unique values. 
**d_first_name**: Data is of type text. There are 1409 unique values. 
**d_mid_name**: Data is of type text. There are 8684 unique values. 
**d_last_name**: Data is of type text. There are 6 unique values. 
**d_suffix**: Data is of type text. There are 12633 unique values. 
**d_birth_date**: Data is of type text. There are 11347 unique values. 
**d_death_date**: Data is of type text. There are 95 unique values. 
**section_id**: Data is of type text. There are 69 unique values. 
**row_num**: Data is of type integer. There are 1642 unique values. Coverage spans from 5.0 to 1479.0. 
**site_num**: Data is of type text. 
**cem_name**: Data is of type text. 
**cem_addr_one**: Data is of type missingdata. 
**cem_addr_two**: Data is of type text. There are 1 unique values. 
**city**: Data is of type text. There are 1 unique values. 
**state**: Data is of type integer. There are 1 unique values. Coverage spans from 2822.0 to 2822.0. 
**zip**: Data is of type text. There are 1 unique values. 
**cem_url**: Data is of type text. There are 1 unique values. 
**cem_phone**: Data is of type text. There are 11 unique values. 
**relationship**: Data is of type text. There are 1428 unique values. 
**v_first_name**: Data is of type text. There are 1022 unique values. 
**v_mid_name**: Data is of type text. There are 8627 unique values. 
**v_last_name**: Data is of type text. There are 6 unique values. 
**v_suffix**: Data is of type text. There are 66 unique values. 
**branch**: Data is of type text. There are 1195 unique values. 
**rank**: Data is of type text. There are 34 unique values. 
**war**: Data is of type missingdata. 
**Temporal coverage** for columns d_suffix with resolution day, covering from 1910-05-18 to 1965-10-23.
**Temporal coverage** for columns d_birth_date with resolution day, covering from 1976-12-27 to 2017-11-17.
- Temporal scope: d_suffix: 1862-12-29 to 2017-03-12 (excluding 2 values outside 1800-2100 (e.g. 03/23/1746, 03/14/1717); most values 1906-06-15 to 1963-01-05; 24 far outside (e.g. 12/09/1922, 09/26/1919)); d_birth_date: 1928-01-04 to 2018-06-26 (excluding 1 values outside 1800-2100 (e.g. 06/14/1781); most values 1977-07-19 to 2018-03-07; 45 far outside (e.g. 06/06/2007, 12/14/2003))
- Geographic scope: city: probably city names (not confirmed)
- Measures and units: row_num: 1 to 1674 (median 240) (of 100% parseable values); state: 2822 to 2822 (median 2822)
- Representative categories: index: John, Robert, William, Joseph, James (2227 distinct values); section_id: 3, 4, 1, 2, 5 (69 distinct values); city: RI (100%)
```

### Description A

This dataset contains information about veteran gravesites in Rhode Island, specifically focusing on individuals interred at the Rhode Island Veteran Memorial Cemetery. It includes details such as names, birth and death dates, military branch, rank, and relationships to other veterans. The dataset features a rich semantic profile, with columns representing personal identifiers, temporal entities, and location data. It can be utilized for research on veteran demographics, historical military service, and cemetery records. The data spans from 1910 to 2017, providing a comprehensive overview of veteran gravesites in the state.

### Description B

This dataset contains information about veteran gravesites in Rhode Island, specifically focusing on individuals interred at the Rhode Island Veteran Memorial Cemetery. It includes details such as names, birth and death dates, military branch, rank, and relationships to other individuals. The dataset spans a temporal coverage from 1862 to 2018, with a geographic focus on Exeter, RI. The data is rich in personal identifiers, with over 1,400 unique first names and 1,200 unique last names. This dataset can be used for genealogical research, historical studies, and honoring the contributions of veterans.

| criterion | A | B |
|---|---|---|
| column confusion (value attributed to the wrong column), count | | |
| statements not supported by the profile, count | | |
| key facts missing (coverage, units, measures) | | |
| readability / usefulness 1-5 | | |
| preferred (A/B/tie) | | |

## c4e2ce61-6199-419f-aeda-795c9b153872

### Profile (original form)

```
The key data profile information for this dataset includes:
**business_id**: Data is of type integer. There are 6253 unique values. Coverage spans from 485.0 to 99681.0. 
**business_name**: Data is of type text. There are 5775 unique values. 
**business_address**: Data is of type text. There are 5673 unique values. 
**business_city**: Data is of type text. There are 1 unique values. 
**business_state**: Data is of type text. There are 1 unique values. 
**business_postal_code**: Data is of type integer. There are 62 unique values. Coverage spans from 94102.0 to 94133.0. 
**business_latitude**: Data is of type float. Coverage spans from 37.715986 to 37.803056. 
**business_longitude**: Data is of type float. Coverage spans from -122.502535 to -122.39377. 
**business_location**: Data is of type geocoordinates. 
**business_phone_number**: Data is of type integer. There are 1931 unique values. Coverage spans from 14155237480.0 to 14155955596.0. 
**inspection_id**: Data is of type text. There are 26663 unique values. 
**inspection_date**: Data is of type text. There are 813 unique values. 
**inspection_score**: Data is of type integer. There are 47 unique values. Coverage spans from 61.0 to 100.0. 
**inspection_type**: Data is of type text. There are 15 unique values. 
**violation_id**: Data is of type text. There are 40210 unique values. 
**violation_description**: Data is of type text. 
**risk_category**: Data is of type text. There are 3 unique values. 
**Neighborhoods**: Data is of type integer. There are 110 unique values. Coverage spans from 5.0 to 108.0. 
**SF Find Neighborhoods**: Data is of type integer. There are 110 unique values. Coverage spans from 5.0 to 108.0. 
**Current Police Districts**: Data is of type integer. There are 10 unique values. Coverage spans from 1.0 to 10.0. 
**Current Supervisor Districts**: Data is of type integer. There are 11 unique values. Coverage spans from 1.0 to 11.0. 
**Analysis Neighborhoods**: Data is of type integer. There are 41 unique values. Coverage spans from 1.0 to 41.0. 
**Temporal coverage** for columns inspection_date with resolution day, covering from 2016-10-18 to 2019-09-16.
**Spatial coverage** for columns business_latitude, business_longitude, with type latlong.
**Spatial coverage** for columns business_location, with type point.
- Identity: 53973 rows, 22 columns
- Temporal scope: inspection_date: 2016-10-04 to 2019-11-28; inspection_date: consecutive time points are mostly one day apart (80% of steps)
- Geographic scope: coordinates within longitude -122.5 to -122.4, latitude 37.72 to 37.8; business_city: probably city names (not confirmed); business_state (state) includes CA; business_postal_code (zip code) includes 94110, 94103, 94102, 94109, 94133 and others (62 distinct)
- Measures and units: business_latitude: 0 to 37.82 (median 37.78); business_longitude: -122.5 to 0 (median -122.4); inspection_score: 45 to 100 (median 87); Neighborhoods: 1 to 117 (median 53)
- Representative categories: business_city: San Francisco (100%)
- Limitations: violation_id: 25% missing
```

### Description A

This dataset contains information on restaurant health inspections in San Francisco, covering 53,973 records from 2016 to 2019. Each entry includes details such as business name, address, inspection date, score, and violation descriptions, allowing for an in-depth analysis of health compliance across various establishments. The semantic profile highlights key aspects like organization identification, location representation, and violation classifications, which are crucial for understanding health standards in the food industry. This dataset can be utilized by health officials, researchers, and the public to monitor food safety practices and identify areas for improvement in restaurant hygiene.

### Description B

This dataset contains information on restaurant health inspections in San Francisco, California, covering 53,973 rows and 22 columns. It includes details such as business names, addresses, inspection dates, scores, and violation descriptions, with temporal coverage from October 4, 2016, to November 28, 2019. The dataset provides valuable insights into food safety practices and compliance, allowing users to assess the health standards of various establishments. It features unique identifiers for businesses and inspections, as well as geographic coordinates for precise location tracking. However, it is important to note that approximately 25% of violation IDs are missing.

| criterion | A | B |
|---|---|---|
| column confusion (value attributed to the wrong column), count | | |
| statements not supported by the profile, count | | |
| key facts missing (coverage, units, measures) | | |
| readability / usefulness 1-5 | | |
| preferred (A/B/tie) | | |

## dcefd0c1-c0b3-4ede-8dbd-e898ce1e2e10

### Profile (original form)

```
The key data profile information for this dataset includes:
**VISN**: Data is of type integer. There are 21 unique values. Coverage spans from 1.0 to 23.0. 
**VISN Name**: Data is of type text. 
**Facility Name**: Data is of type text. There are 78 unique values. 
**City**: Data is of type text. There are 136 unique values. 
**Station**: Data is of type integer. There are 128 unique values. Coverage spans from 402.0 to 693.0. 
**ST**: Data is of type text. There are 52 unique values. 
**Station.1**: Data is of type text. There are 139 unique values. 
**Complexity**: Data is of type text. There are 6 unique values. 
**Inpatient Number of Patients African American**: Data is of type missingdata. 
**Inpatient Patient Satisfacion African American**: Data is of type float. Coverage spans from 63.7 to 89.0. 
**Inpatient Number of Patients White**: Data is of type missingdata. 
**Inpatient Patient Satisfaction White**: Data is of type float. Coverage spans from 54.6 to 94.1. 
**Inpatient Number of Patients Other**: Data is of type missingdata. 
**Inpatient Patient Satisfaction Other**: Data is of type float. Coverage spans from 42.1 to 89.9. 
**Outpatient Number of Patients African American**: Data is of type missingdata. 
**Outpatient Patient Satisfacion African American**: Data is of type float. Coverage spans from 71.5 to 87.5. 
**Outpatient Number of Patients White**: Data is of type missingdata. 
**Outpatient Patient Satisfaction White**: Data is of type float. Coverage spans from 31.8 to 92.9. 
**Outpatient Number of Patients Other**: Data is of type missingdata. 
**Outpatient Patient Satisfaction Other**: Data is of type float. Coverage spans from 50.1 to 97.9. 

- Geographic scope: City: probably city names (not confirmed); ST (state) includes NY, CA, PA, TX, FL and others (52 distinct)
- Measures and units: VISN: 1 to 23 (median 11); Station: 402 to 757 (median 590); Inpatient Patient Satisfacion African American: 63.7 to 89 (median 80.1); Inpatient Patient Satisfaction White: 54.6 to 94.1 (median 75.8)
- Representative categories: VISN Name: VA Stars & Stripes Healthcare Network - VISN 4 (7% of 139 non-blank), South Central VA Health Care Network - VISN 16, VA New England Health Care System - VISN 1, VA Southeast Network - VISN 7, Northwest Network - VISN 20 (21 distinct values); Complexity: 3 (27% of 139 non-blank), 2, 1a, 1c, 1b (6 distinct values)
```

### Description A

This dataset focuses on Veteran Patient Satisfaction, providing insights into the experiences of patients across various healthcare facilities within the Veterans Affairs system. It includes information on patient demographics, satisfaction ratings, and the complexity of care at different facilities. The dataset features 21 unique VISNs and 78 distinct facility names, covering a wide range of cities and states. Each column represents specific aspects of patient care, such as inpatient and outpatient satisfaction scores for different demographic groups. This dataset can be utilized to analyze trends in patient satisfaction among veterans, identify areas for improvement, and inform healthcare policy decisions.

### Description B

This dataset focuses on Veteran Patient Satisfaction, providing insights into healthcare experiences across various facilities in the United States. It includes data on patient demographics, satisfaction ratings, and facility complexity, covering 21 unique VISNs and 78 facilities in cities across multiple states. The dataset captures inpatient and outpatient satisfaction scores for African American, White, and other patient groups, with satisfaction ratings ranging from 42.1 to 94.1. This information can be used to analyze healthcare quality and patient experiences, helping to identify areas for improvement in veteran healthcare services.

| criterion | A | B |
|---|---|---|
| column confusion (value attributed to the wrong column), count | | |
| statements not supported by the profile, count | | |
| key facts missing (coverage, units, measures) | | |
| readability / usefulness 1-5 | | |
| preferred (A/B/tie) | | |

## dde84e96-8559-4a9c-9b96-d349f069dd65

### Profile (original form)

```
The key data profile information for this dataset includes:
**Lobbyist Name**: Data is of type text. There are 10 unique values. 
**Filing**: Data is of type text. There are 68 unique values. 
**Report Type**: Data is of type text. 
**Date Filed**: Data is of type text. There are 50 unique values. 
**Period Start Date**: Data is of type text. There are 28 unique values. 
**Period End Date**: Data is of type text. There are 31 unique values. 
**Activity**: Data is of type text. There are 1 unique values. 
**Termination**: Data is of type text. There are 2 unique values. 
**Amendment Filing**: Data is of type text. There are 2 unique values. 
**Amendment to Report Filed**: Data is of type text. There are 6 unique values. 
**Type of Expenditure Lobbyist**: Data is of type text. There are 4 unique values. 
**Total Amount Spent to Influence**: Data is of type float. Coverage spans from 0.0 to 125454.15. 
**Temporal coverage** for columns Date Filed with resolution day, covering from 2016-06-17 to 2018-12-13.
**Temporal coverage** for columns Period Start Date with resolution week, covering from 2016-06-01 to 2018-08-01.
**Temporal coverage** for columns Period End Date with resolution day, covering from 2016-06-17 to 2018-08-31.
**Temporal coverage** for columns Amendment to Report Filed with resolution month, covering from 2016-07-26 to 2018-03-16.
- Identity: 70 rows, 12 columns
- Temporal scope: Date Filed: 2016-06-17 to 2018-12-13; Period Start Date: 2016-06-01 to 2018-11-01; Period End Date: 2016-06-17 to 2018-11-30
- Measures and units: Total Amount Spent to Influence: 0 to 125454 (median 10376)
- Representative categories: Lobbyist Name: Mission for All, LLC (17%), Richard Drury, Airbnb, Inc., RAI Services Company, San Francisco Community Alliance for Jobs & Housing (10 distinct values); Report Type: Disclosure Form SFEC 2110B (86%), Registration Form SFEC 2110A; Activity: True (100% of 33 non-blank); Termination: True (71% of 7 non-blank), False
- Limitations: Activity: 53% missing; Termination: 90% missing; Amendment Filing: 91% missing
```

### Description A

This dataset contains Lobbyist Expenditure Reports, featuring 70 entries that document lobbying activities from June 17, 2016, to December 13, 2018. It includes key information such as the names of lobbyists, types of reports, filing dates, and total amounts spent to influence legislation. The dataset is structured with columns representing various aspects, including organization names, document types, and monetary values. It can be used to analyze lobbying trends, assess financial influence in politics, and understand the activities of different organizations. The semantic profiles of the columns enhance the dataset's usability for research and regulatory purposes.

### Description B

This dataset contains 70 rows and 12 columns of Lobbyist Expenditure Reports, covering the period from June 17, 2016, to December 13, 2018. It includes information on various lobbyists, such as their names, types of filings, and the total amounts spent to influence legislation, with values ranging from 0 to 125,454. The dataset features 10 unique lobbyist names and 68 unique filings, primarily consisting of Disclosure Form SFEC 2110B. It can be used to analyze lobbying activities and expenditures, providing insights into political influence and financial contributions in the lobbying sector. However, some fields, such as Activity and Termination, have significant missing data.

| criterion | A | B |
|---|---|---|
| column confusion (value attributed to the wrong column), count | | |
| statements not supported by the profile, count | | |
| key facts missing (coverage, units, measures) | | |
| readability / usefulness 1-5 | | |
| preferred (A/B/tie) | | |

## 14f56ea0-2392-45ef-8236-7e569e8907bc

### Profile (original form)

```
The key data profile information for this dataset includes:
**Election Date**: Data is of type text. There are 5 unique values. 
**City Elective Office**: Data is of type text. There are 8 unique values. 
**Amount of VEC**: Data is of type integer. There are 2 unique values. Coverage spans from 104000.0 to 243000.0. 
**Status of VEC**: Data is of type text. 
**Temporal coverage** for columns Election Date with resolution year, covering from 2012-11-06 to 2016-11-08.
- Identity: 16 rows, 4 columns
- Temporal scope: Election Date: 2012-11-06 to 2016-11-08
- Geographic scope: City Elective Office: probably city names (not confirmed)
- Measures and units: Amount of VEC: 104000 to 243000 (median 243000)
- Representative categories: City Elective Office: Community College Board (25%), Board of Education, City Attorney, Treasurer, Assessor (8 distinct values); Amount of VEC: 243000.0 (56%), 104000.0; Status of VEC: The ceiling is not in place. (81%), The ceiling is in place., The Ceiling is in place., The ceiling is lifted. On June 27, 2012, the Ethics Commission lifted the VEC when a candidate for the College Board reported receiving contributions in excess of the VEC.
```

### Description A

This dataset focuses on Voluntary Expenditure Ceilings (VEC) and includes 16 entries detailing election dates, city elective offices, amounts of VEC, and their statuses from November 6, 2012, to November 8, 2016. The dataset captures various political positions, with the Community College Board and Board of Education being the most frequently represented. The monetary values range from 104,000 to 243,000, with a significant majority indicating that the ceiling is not in place. This dataset can be used to analyze campaign finance regulations and the impact of VEC on electoral processes. Each column is structured to represent temporal entities, positions, monetary values, and their statuses, providing a comprehensive view of VEC dynamics in the political landscape.

### Description B

This dataset contains 16 rows and 4 columns, focusing on Voluntary Expenditure Ceilings (VEC) for various city elective offices from November 6, 2012, to November 8, 2016. It includes information on the election date, the specific city elective office, the monetary amount of the VEC, and the status of the VEC. The dataset features 8 unique city elective offices and two distinct monetary values, ranging from 104,000 to 243,000. The status of the VEC indicates whether the ceiling is in place or has been lifted, providing insights into campaign finance regulations and their implications for electoral candidates.

| criterion | A | B |
|---|---|---|
| column confusion (value attributed to the wrong column), count | | |
| statements not supported by the profile, count | | |
| key facts missing (coverage, units, measures) | | |
| readability / usefulness 1-5 | | |
| preferred (A/B/tie) | | |

## 1cf07054-8055-4120-9d79-eaa6a1b4e08f

### Profile (original form)

```
The key data profile information for this dataset includes:
**Month**: Data is of type text. There are 12 unique values. 
**Sort**: Data is of type integer. There are 12 unique values. Coverage spans from 1.0 to 12.0. 
**Year**: Data is of type text. There are 23 unique values. 
**Page Views**: Data is of type integer. There are 266 unique values. Coverage spans from 11632495.0 to 310159256.0. 
**Page Visits**: Data is of type text. There are 240 unique values. 
**Temporal coverage** for columns Year with resolution year, covering from 2003-01-01 to 2025-01-01.
- Identity: 267 rows, 5 columns
- Temporal scope: Year: 2003 to 2025; Year: consecutive time points are one year apart
- Measures and units: Sort: 1 to 12 (median 7); Page Views: 318355 to 1075594920 (median 70644944); Page Visits: 40122 to 464305872 (median 22287094)
- Representative categories: Month: July (9%), August, September, February, March (12 distinct values); Sort: 7 (9%), 8, 9, 2, 3 (12 distinct values)
```

### Description A

This dataset, titled "CDC Page Views," contains 267 rows and 5 columns, capturing web analytics data from 2003 to 2025. It includes monthly records of page views and page visits, providing insights into user engagement with CDC content. The columns represent temporal entities, with "Month" and "Year" indicating the time of data collection, while "Page Views" and "Page Visits" serve as key metrics for measuring web traffic. This dataset can be utilized for analyzing trends in public health information access, evaluating the effectiveness of CDC outreach efforts, and understanding user behavior over time.

### Description B

This dataset, titled "CDC Page Views," contains 267 rows and 5 columns, capturing web analytics data from 2003 to 2025. It includes monthly page views and visits, with data organized by month and year. The dataset features 12 unique months and 23 unique years, providing a comprehensive temporal coverage. The Page Views and Page Visits columns serve as key metrics for web traffic analysis, allowing for aggregation and measurement of user engagement. This dataset can be utilized for understanding trends in public health information access and evaluating the effectiveness of CDC outreach efforts over time.

| criterion | A | B |
|---|---|---|
| column confusion (value attributed to the wrong column), count | | |
| statements not supported by the profile, count | | |
| key facts missing (coverage, units, measures) | | |
| readability / usefulness 1-5 | | |
| preferred (A/B/tie) | | |

## 2a208933-c2c5-4f23-9c63-4771358d199d

### Profile (original form)

```
The key data profile information for this dataset includes:
**Year**: Data is of type text. There are 7 unique values. 
**Month Number**: Data is of type integer. There are 12 unique values. Coverage spans from 1.0 to 12.0. 
**Month**: Data is of type text. There are 12 unique values. 
**Utility**: Data is of type text. There are 4 unique values. 
**Owner**: Data is of type text. There are 3 unique values. 
**Units**: Data is of type text. There are 4 unique values. 
**Usage**: Data is of type float. Coverage spans from 12.766116 to 15358398.0. 
**Temporal coverage** for columns Year with resolution year, covering from 2013-01-01 to 2019-01-01.
- Identity: 560 rows, 7 columns
- Temporal scope: Year: 2013 to 2019; Year: consecutive time points are one year apart
- Measures and units: Usage: 10.91 to 16421214 (median 254967); units listed in Units: Therms, kWh, Million Gallons, PAX
- Representative categories: Month Number: 1 (9%), 2, 3, 4, 5 (12 distinct values); Month: Jan (9%), Feb, Mar, Apr, May (12 distinct values); Utility: Gas (29%), Electricity, Water, Passengers; Owner: Commission (43%), Tenant, Campus
```

### Description A

This dataset on Airport Utility Consumption contains 560 rows and 7 columns, capturing utility usage data from 2013 to 2019. It includes measurements of Electricity, Water, Gas, and Passengers, with usage recorded in various units such as kWh, Million Gallons, and Therms. The dataset features a structured profile with unique values for each column, representing temporal entities like Year and Month, as well as utility types and measurement units. This dataset can be utilized for analyzing trends in utility consumption at airports, aiding in resource management and sustainability efforts.

### Description B

This dataset contains 560 rows and 7 columns, focusing on airport utility consumption from 2013 to 2019. It includes measurements of electricity, water, gas, and passenger counts, with usage recorded in various units such as kWh, Million Gallons, and Therms. The data is organized by year and month, providing a temporal resolution that allows for analysis of trends over time. This dataset can be used to assess utility consumption patterns at airports, aiding in resource management and sustainability efforts. It features contributions from different owners, including tenants and commissions, reflecting diverse operational contexts.

| criterion | A | B |
|---|---|---|
| column confusion (value attributed to the wrong column), count | | |
| statements not supported by the profile, count | | |
| key facts missing (coverage, units, measures) | | |
| readability / usefulness 1-5 | | |
| preferred (A/B/tie) | | |
