const {db,genid} = require('../db/dbUtils')

/**
 * Examine whether the user is login
 * @param {*} req 
 * @param {*} res 
 * @param {*} next 
 * @returns 
 */
async function authMiddleware(req, res, next) {
    const account = req.header('account');
    const token = req.header('token');
    if (!account || !token) {
        return res.status(401).json({ error: 'Unauthorized' });
    }
    try {
        const { err, rows } = await db.async.all(
            'SELECT * FROM admin WHERE account = ? AND token = ?',
            [account, token]
        );
        if (rows.length === 0) {
            return res.status(401).json({ error: 'Unauthorized' });
        }
        req.user = rows[0];
        next();
    } catch (err) {
        res.status(500).json({ error: 'Auth check failed' });
    }
}

module.exports = authMiddleware;