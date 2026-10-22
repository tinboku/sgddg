# Held-out atomic applicability checklist

Source: the 17 raw CSVs named in `heldout_ids.txt`. Each item is an atomic, source-backed assertion. Labels are `[core]` for time/grain, entity/geography, or main measure/unit facts and `[caveat]` for applicability limitations such as mixed levels, missingness, placeholders, suppressed values, or non-comparable units. This checklist contains 68 core assertions and 17 caveat assertions.

## 0cbb09dc-4012-4f18-900e-6d746bfc10d3

1. [core] The table covers 2023 Q1 through 2025 Q4.
2. [core] The `Time Period` field is `12 Month-ending` for every row.
3. [core] The table contains two topics: `Rates by Age` and `Rates by Cause`.
4. [core] Rates use two units: per 1,000 births and per 100,000 births.
5. [caveat] `Rate` is not directly comparable across rows without conditioning on the `Unit` and `Indicator`.

## 10811fc1-8ad0-4d8b-bdcd-cde9b43c86e6

1. [core] The table has one filing-record row per `rep_file_num` (925 distinct values among 926 rows).
2. [core] `filing_date` values span 2006–2018.
3. [core] The entities are investment companies or funds identified by `entity_name` and CIK.
4. [core] The records include filing type in `doc_type_code` and issuer location in city/state/ZIP fields.
5. [caveat] State codes include nonstandard or non-US-looking values and ZIP values include Canadian-style codes, so location fields should not be assumed to be US-only.

## 14f56ea0-2392-45ef-8236-7e569e8907bc

1. [core] The table contains 16 rows of city elective-office records.
2. [core] `Election Date` covers elections from 2012 through 2016.
3. [core] There are eight elective-office categories, including Assessor, Sheriff, Treasurer, and board offices.
4. [core] `Amount of VEC` is either 104,000.00 or 243,000.00.
5. [caveat] `Status of VEC` has inconsistent capitalization and wording, including `The Ceiling is in place.` and `The ceiling is in place.`.

## 1bf0f2e9-ee0d-4d99-8d34-e4fbf59fd8b3

1. [core] Each row has a unique `Eviction ID`.
2. [core] The records describe eviction notices in California, with San Francisco as the dominant city.
3. [core] `File Date` spans 1997–2026.
4. [core] Cause fields such as `Non Payment`, `Breach`, `Nuisance`, and `Owner Move In` are boolean flags.
5. [caveat] Neighborhood, district, location, and many derived geography fields are blank for a substantial share of rows, so a missing geography is not evidence of no geography.

## 1cf07054-8055-4120-9d79-eaa6a1b4e08f

1. [core] The table records monthly page views and page visits.
2. [core] `Year` spans 2003–2025.
3. [core] `Month` is accompanied by a numeric month-order field `Sort`.
4. [core] The main measures are `Page Views` and `Page Visits`.
5. [caveat] The time series is incomplete: not every year has all twelve months, so annual comparisons require checking month coverage.

## 1ff67fc2-ff44-48ea-8b9b-d19bc71c9e08

1. [core] The table contains 78,300 bank or financial-institution records.
2. [core] Each record identifies an institution with fields such as `NAME`, `UNINUM`, `CITY`, `STALP`, and `ZIP`.
3. [core] Institution class is represented by `BKCLASS`, and service type by `SERVTYPE`.
4. [core] The file includes establishment and run-date fields (`ESTYMD` and `RUNDATE`).
5. [caveat] CBSA, division, CSA, and other geography fields are optional classification fields with substantial missingness; they should not be treated as a complete geographic hierarchy.

## 201460a5-ac75-422c-99ec-3bceef8d195c

1. [core] Each row has a unique `DR Number` crime-report identifier.
2. [core] `Date Reported` and `Date Occurred` provide report and occurrence dates, with records spanning 2010–2024.
3. [core] The table identifies 21 reporting areas through `Area ID`/`Area Name`.
4. [core] `Crime Code` and `Crime Code Description` are fixed in this extract, while victim and premise fields describe the reported incidents.
5. [caveat] `MO Codes`, victim demographics, and location fields can be blank or multi-valued, so a row should not be interpreted as a complete demographic or modus-operandi profile.

## 2a3cc123-f9f6-4e1d-a03f-3e33056e5021

1. [core] The table covers years 1970–2015.
2. [core] Rows are stratified by six age groups from 15–19 years through 40–44 years.
3. [core] Rows are also stratified by six race or Hispanic-origin categories.
4. [core] The main measure is `Birth Rate`.
5. [caveat] Birth-rate values are subgroup-specific and cannot be compared across rows without holding both age group and race/origin category fixed.

## 2f1aabc5-904c-4bad-90ba-0e5e949589e8

1. [core] The table records one emergency call per unique `NOPD_Item`.
2. [core] `TimeCreate`, `TimeDispatch`, `TimeArrival`, and `TimeClosed` represent the call lifecycle.
3. [core] Calls are classified by type, priority, disposition, beat, ZIP, and police district.
4. [core] The main outcome field is `DispositionText`, with ten observed disposition categories.
5. [caveat] Dispatch, arrival, and coordinate fields contain missing values and zero-like coordinate placeholders, so elapsed times and mapped locations are not available for every call.

## 4e4c5dbb-fcd4-4718-afb5-8e11b5b90413

1. [core] The table contains ten workforce-program metrics identified by `Metric Nbr`.
2. [core] Each row compares an annual goal with second-quarter cumulative goal and actual fields.
3. [core] The table includes percentage attainment for the second-quarter goal and annual goal completion.
4. [core] Metrics mix counts, placements, and wage measures, including average wage at placement.
5. [caveat] The goal and actual columns mix units and contain blanks for some metrics, so percentages must not be aggregated across all rows.

## 53acdac4-4f73-44fa-ad63-bdfad92472d6

1. [core] The table covers 51 states and county-level records.
2. [core] The geographic grain is census tract within county, identified by `Census Tract Number`.
3. [core] The main measure is `Life Expectancy`.
4. [core] Each life-expectancy value is accompanied by a range and a standard error field.
5. [caveat] Some rows have blank life-expectancy, range, and standard-error values, so missingness must be distinguished from a measured value.

## 5f04aaa3-5cd9-47af-a159-6412ab19375d

1. [core] The table records one call per unique `NOPD_Item`.
2. [core] Call creation dates run from 2017 onward in the extract.
3. [core] Calls include initial and final type, initial priority, disposition, and self-initiated status.
4. [core] The table provides beat, ZIP, police district, and a location field for calls.
5. [caveat] `TimeDispatch` and `TimeArrive` are missing for many rows and `MapX`/`MapY` include zero placeholders, limiting response-time and spatial interpretations.

## 62acddae-6081-4f24-b3ad-b69877c0a8d9

1. [core] The table covers annual records from 2003 through 2020.
2. [core] The geographic grain is county within state, identified by FIPS fields.
3. [core] The main measure is county birth rate.
4. [core] Each birth-rate observation includes lower and upper confidence limits.
5. [caveat] County FIPS values are not one-to-one with displayed county names in every row, so geography should be checked using the combined FIPS fields rather than names alone.

## 7b64babd-8c66-4dcb-9be7-6bcd2ef35e04

1. [core] The table contains five geographic rows: four states and the United States.
2. [core] The two measure columns are labeled 2012 and 2014.
3. [core] The values in those columns are rates for the corresponding geography and year.
4. [core] The `Location` field embeds the geography name and coordinates.
5. [caveat] The table mixes state-level rows with a United States aggregate, so the national row must not be treated as another state.

## c4ca6131-444b-4ffe-90cf-1482709f62b7

1. [core] The table is a lookup of 55 `ExceptionCode` values.
2. [core] Each exception code maps to one explanatory `Message`.
3. [core] Messages describe construction, excavation, paving, utility, traffic, or coordination constraints.
4. [core] This is a code-to-description reference table rather than an event table with dates or quantities.
5. [caveat] Messages contain operational instructions and contact details, so they should be interpreted as exception guidance rather than measured outcomes.

## dde84e96-8559-4a9c-9b96-d349f069dd65

1. [core] The table contains lobbying registration and disclosure filings.
2. [core] `Date Filed` records filing dates spanning 2016–2018.
3. [core] Filing rows are categorized as registration or disclosure forms.
4. [core] The table identifies lobbyist name, expenditure-lobbyist type, reporting period, and amount spent to influence.
5. [caveat] Amounts are blank for many registration rows and some disclosures, so blank spending is not equivalent to zero spending.

## f359e457-7d8c-46ee-876a-abb5340e1168

1. [core] The table contains state-by-month observations for 2015–2026.
2. [core] The `Period` field is `12 month-ending` for all rows.
3. [core] Rows are stratified by state and drug-related `Indicator`.
4. [core] The main measure is `Data Value`, with percent-complete and pending-investigation fields.
5. [caveat] Many values are suppressed or blank and footnotes explicitly flag low data quality or suppression, so missing `Data Value` must not be read as zero.
