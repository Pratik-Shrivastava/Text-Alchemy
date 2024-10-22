INSERT INTO roles
(name, description)
VALUES
('SUPER_ADMIN', 'Has all access and controls over the entire system, including managing other admins.'),
('ADMIN', 'Has administrative access, can manage users and perform administrative tasks, but with limited control compared to SUPER_ADMIN.'),
('CUSTOMER', 'Basic user role with access to system features designated for end users.');

INSERT INTO users (role_id, first_name, last_name, email, active)
VALUES (1, 'Pratik', 'Shrivastava', 'pratikkumar441@gmail.com', TRUE);

