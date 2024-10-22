SQL - Structured Query Language
=================================

| Terminology |	Description |
| :------------------------------------- | :---------------------------------------------- |
| SQL	| SQL is a `programming language` used to manage and manipulate data stored in a relational database management system (RDBMS). |
| SQL Commands	| Four types: (i) DDL (Data Definition Language), (ii) DML (Data Manipulation Language), (iii) DCL (Data Control Language), and (iv) TCL (Transaction Control Language) |
| DDL vs DML	| DDL provides the ability to define, create and modify database objects such as tables, views, indexes, and users. DML allows for manipulating data in a database, such as inserting, updating, and deleting records. |
| Query Plan	| A query plan is a set of steps that the database engine uses to execute a SQL statement. |
| Primary Key	| It is a column or set of columns that uniquely identifies each row in a table. |
| Foreign Key	| It is a column or set of columns that refer to the primary key of another table. It is used to establish a relationship between two tables. |
| Primary Key vs Foreign Key	| i) A primary key is a column or set of columns that uniquely identifies each row in a table. ii) A foreign key is a column or set of columns that refer to the primary key of another table, establishing a relationship between the two tables. |
| Normalization	| It is the `process` of organizing data in a database to minimize redundancy and dependency. It involves splitting large tables into smaller ones and creating relationships between them to reduce data duplication. |
| De-normalization	| De-normalization is the `process` of adding redundant data to a database to improve performance or simplify queries. It involves duplicating data in one or more tables to avoid joining multiple tables in a query. |
| Stored Procedure	| It is a precompiled set of `SQL statements` that can be called from an application. |
| Trigger	| It is a special type of `SQL statement` (stored procedure) and automatically executed in response to certain database events (INSERT, UPDATE or DELETE operation). |
| Cursor	| A cursor is a `database object` that allows you to retrieve and manipulate rows of data one at a time. Cursors are often used in stored procedures and trigger to process data sequentially. |
| Function	| A function is a `set of SQL statements` that are stored in the database and can be executed repeatedly. |
| Stored Function	| A stored function is a `program` that performs a specific task and returns a value. |
| Join	| A join is a `SQL operation` that combines rows from two or more tables based on a related column between them. There are different types of joins, including INNER JOIN, LEFT JOIN, RIGHT JOIN, and FULL OUTER JOIN. |
| Self-join	| It is a join between a table and itself and it is used to retrieve data from the table itself and often used to find relationships between rows within the same table. |
| Left-join	| It is a type of join that returns all rows from the left table and matching rows from the right table. |
| Right-join	| It is a type of join that returns all rows from the right table and matching rows from the left table. |
| Inner-join	| An inner join returns only the rows that have matching values in both tables |
| Outer-join	| An outer join returns all the rows from one table and the matching rows from the other table. |
| Full Outer-join	| It is a type of join that returns all rows from both the left and right tables, including any rows that do not have matching values in the other table. |
| View	| It is a `virtual table` created by a SQL query. The result of a SELECT statement. |
| View vs Table	| i) A table is a physical object that stores data, while a view is a virtual object that represents a SELECT statement. ii) Views are used to simplify queries and to provide a layer of abstraction between the database and the user. |
| Database vs Schema	| i) A database is a collection of related data that is stored on a computer. ii) Schema is a logical container for database objects, such as tables, views, and stored procedures. iii) A database can contain multiple schemas, and a schema can be used to organize and manage objects within a database. |
| Subquery	| It is a SQL query nested within another query. |
| Correlated Subquery	| A correlated subquery is a subquery that depends on the outer query for its values. |
| Index	| It is a database object that is used to improve query performance by allowing the database engine to quickly locate data in a table. |
| Clustered Index	| A clustered index is an index that determines the physical order of data in a table. It is created on the primary key of the table and is used to improve query performance by reducing the number of disk I/O operations required to retrieve data. It is representing the main data. A table can have only one clustered index. |
| Non-clustered Index	| It is an index created on one/more columns of a table that quickly find rows based on the values in the indexed columns during query. It is representing copy of the main data. A table can have multiple non-clustered indexes. |
| Clustered Index vs Non-clustered Index	| i) A clustered index determines the physical order of data in a table and is created on the primary key or a unique column. A non-clustered index is created on a non-primary key column and does not affect the physical order of data. ii) A table can have only one clustered index, it can have multiple non-clustered indexes. |
| Transaction	| A transaction is a sequence of SQL statements that are treated as a single unit of work. |
| Data Integrity	| Data integrity refers to the accuracy and consistency of data in a database. Constraints, triggers, and referential integrity are such examples. Common types of constraints include primary key, foreign key, unique, check, and not null constraints. |
| Union	| It is an operation that combines the results of two or more SELECT statements into a single result set. |
| Union vs Union All	| UNION removes duplicate rows, while UNION ALL does not. |
| Where vs Having	| i) WHERE is used to filter rows based on a condition, while HAVING is used to filter groups based on a condition. ii) WHERE is used with SELECT, UPDATE, and DELETE statements, while HAVING is used with SELECT statements that include a GROUP BY clause. |
| Deadlock	| It is a situation where two or more transactions are blocked. Waiting for each other to release resources. |
| Temporary Table	| A temporary table is a table that is created and exists only for the duration of a session or transaction. |
