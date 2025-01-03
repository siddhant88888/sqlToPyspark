CREATE PROCEDURE increaseSalary()
BEGIN
    UPDATE employees SET salary = salary * 1.10 WHERE department = 'Engineering';
END;