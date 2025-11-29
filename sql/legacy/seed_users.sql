-- Seed file for Users table
-- This file creates the default admin user for production deployment
-- 
-- IMPORTANT: Change the password after first login in production!
-- 
-- Default Credentials:
-- Email: admin@mastergroup.com
-- Password: MG@2024#Secure!Pass
--
-- To generate a new password hash, use Python:
-- import bcrypt
-- password = b"your_password_here"
-- hashed = bcrypt.hashpw(password, bcrypt.gensalt())
-- print(hashed.decode())

-- Insert default admin user
INSERT INTO users (email, password_hash, full_name, is_active) 
VALUES (
    'admin@mastergroup.com',
    '$2b$12$8ALlBQw1UrHePD2QyYRy0uGz/mMEOsay4HzCwvPjMt8nOmGlQ/8MO',
    'Admin User',
    true
)
ON CONFLICT (email) DO UPDATE SET
    password_hash = EXCLUDED.password_hash,
    full_name = EXCLUDED.full_name,
    is_active = EXCLUDED.is_active;

-- Verify insertion
SELECT 
    id,
    email,
    full_name,
    is_active,
    created_at,
    'Password: MG@2024#Secure!Pass' as note
FROM users 
WHERE email = 'admin@mastergroup.com';

-- Display success message
DO $$
BEGIN
    RAISE NOTICE '✅ Admin user seeded successfully!';
    RAISE NOTICE 'Email: admin@mastergroup.com';
    RAISE NOTICE 'Password: MG@2024#Secure!Pass';
    RAISE NOTICE '';
    RAISE NOTICE '⚠️  IMPORTANT: Change this password after first login in production!';
    RAISE NOTICE '';
    RAISE NOTICE 'To change password:';
    RAISE NOTICE '1. Generate hash: python -c "import bcrypt; print(bcrypt.hashpw(b\"newpass\", bcrypt.gensalt()).decode())"';
    RAISE NOTICE '2. Update: UPDATE users SET password_hash = ''hash'' WHERE email = ''admin@mastergroup.com'';';
END $$;
