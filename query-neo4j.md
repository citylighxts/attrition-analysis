# Query Neo4j
Load CSV + add EmployeeID berdasar row number

```
LOAD CSV WITH HEADERS FROM 'https://raw.githubusercontent.com/citylighxts/analisis-attrition/refs/heads/main/final_employee_data.csv' AS row
MERGE (e:Employee {EmployeeID: toInteger(row.EmployeeID)})
SET  e.Age = toInteger(row.Age),
     e.MonthlyIncome = toInteger(row.MonthlyIncome),
     e.TotalWorkingYears = toInteger(row.TotalWorkingYears),
     e.OverTime = row.OverTime,
     e.Department = row.Department,
     e.JobRole = row.JobRole,
     e.BusinessTravel = row.BusinessTravel,
     e.MaritalStatus = row.MaritalStatus,
     e.DistanceFromHome = toInteger(row.DistanceFromHome),
     e.Education = toInteger(row.Education),
     e.EducationField = row.EducationField,
     e.TotalSatisfaction = toInteger(row.TotalSatisfaction),
     e.CareerStability = toFloat(row.CareerStability),
     e.LoyaltyRatio = toFloat(row.LoyaltyRatio),
     e.IncomePerAge = toFloat(row.IncomePerAge),
     e.AttritionRisk = toFloat(row.AttritionRisk),
     e.Prediction = toInteger(row.Prediction);
```
Membuat node Department
```
MATCH (e:Employee)
WITH DISTINCT e.Department AS dept
MERGE (:Department {name: dept});
```
Membuat node JobRole
```
MATCH (e:Employee)
WITH DISTINCT e.JobRole AS role
MERGE (:JobRole {name: role});
```
Membuat relasi WORKS_IN

```
MATCH (e:Employee)
MATCH (d:Department {name: e.Department})
MERGE (e)-[:WORKS_IN]->(d);
```
Membuat relasi HAS_ROLE
```
MATCH (e:Employee)
MATCH (r:JobRole {name: e.JobRole})
MERGE (e)-[:HAS_ROLE]->(r);
```
List 10 Employee dengan AttritionRisk tertinggi

```
MATCH (e:Employee)
RETURN e.EmployeeID, e.JobRole, e.Department, e.AttritionRisk
ORDER BY e.AttritionRisk DESC
LIMIT 10;
```


Jumlah pegawai tiap departemen yang memiliki High AttritionRisk

Berdasarkan model, threshold-nya adalah 0.279

```
MATCH (e:Employee)-[:WORKS_IN]->(d:Department)
WHERE e.AttritionRisk >= 0.279
RETURN d.name AS Department,
       COUNT(e) AS HighRiskEmployees
ORDER BY HighRiskEmployees DESC;
```

List pegawai yang memiliki AttritionRisk tinggi

```
MATCH (e:Employee)-[:WORKS_IN]->(d:Department)
MATCH (e)-[:HAS_ROLE]->(r:JobRole)
WHERE e.AttritionRisk >= 0.279
RETURN d.name AS Department,
       r.name AS JobRole,
       e.EmployeeID,
       e.AttritionRisk
ORDER BY e.AttritionRisk DESC;
```

Rata-rata AttritionRisk karyawan yang lembur dan tidak lembur

```
MATCH (e:Employee)
WITH e.OverTime AS OT, COUNT(*) AS total, AVG(e.AttritionRisk) AS avgRisk
RETURN OT, total, avgRisk
ORDER BY avgRisk DESC;
```


Role dengan AttritionRisk tertinggi
```
MATCH (e:Employee)-[:HAS_ROLE]->(j:JobRole)
RETURN j.name AS JobRole,
       COUNT(e) AS TotalEmployees,
       AVG(e.AttritionRisk) AS AvgRisk
ORDER BY AvgRisk DESC;
```