// jshint esversion:6

require('dotenv').config();

// const createError = require('http-errors');
const express = require('express');
const path = require('path');
// const assert = require('assert');
const bodyParser = require("body-parser");
// const cookieParser = require('cookie-parser');
// const flash = require('connect-flash');
const ejs = require('ejs');
const flash = require('express-flash')

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const bcrypt = require('bcrypt');
const saltRounds = 12;
const salt = process.env.SALT;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const sanitizeHtml = require('sanitize-html');

const app = express();

app.use(express.static('public'));
app.set('view engine', 'ejs');
app.use(bodyParser.urlencoded({extended: true}));

const _ = require('lodash');
const moment = require('moment');

const fs = require('fs');

// Debugging with DevTools inspector
const inspector = require('inspector');
const inspectorSession = new inspector.Session();
inspectorSession.connect();
inspectorSession.post('Profiler.enable');
inspectorSession.post('Profiler.start');
// setTimeout(_ => {
//     inspectorSession.post('Profiler.stop', (err, data) => {
//         fs.writeFileSync('data.cpuprofile', JSON.stringify(data.profile));
//     });
// }, 3000);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Use PassportJs for cookies and session management

const session = require('express-session');
const passport = require("passport");
const passportLocalMongoose = require("passport-local-mongoose");
const GoogleStrategy = require('passport-google-oauth20').Strategy;
const findOrCreate = require('mongoose-findorcreate');

app.use(session({
    secret: process.env.SALT,
    resave: false,
    saveUninitialized: false,
    // cookie: { secure: false, maxAge: 14400000 }
}));
app.use(flash());
app.use(passport.initialize());
app.use(passport.session());

app.use((req, res, next) => {
    res.locals.isAuthenticated = req.isAuthenticated();
    next();
});

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Load properties from config.json
const { webapp_host, webapp_port, 
    local_db_username, local_db_password, local_db_server, local_db_server_port, 
    cloud_db_username, cloud_db_server, cloud_db_pem_file, cloud_db_password, cloud_db_server_port, 
    db_name } = require(`${__dirname}/config.json`);

// Heroku cloud deployment uses random port in "process.port.PORT" environment variable
let herokuPort = webapp_port;
if (process.env.PORT != null && process.env.PORT != "") {
    herokuPort = process.env.PORT;
}


// Define or read default configs, then overrides from ENV vars defined at OS level
// let config = {
//     APP_DB_HOST: "3.82.161.206",
//     APP_DB_USER: "nodeapp",
//     APP_DB_PASSWORD: "coffee",
//     APP_DB_NAME: "COFFEE"
// }  
// Object.keys(config).forEach(key => {
//     if(process.env[key] === undefined) {
//         console.log(`[NOTICE] Value for key '${key}' not found in ENV, using default value. See config.js`);
//     } else {
//         config[key] = process.env[key];
//     }});
// module.exports = config;

  
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const mongoose = require('mongoose');

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

if (process.env.USE_CLOUD_DB === 'true') {
    // Set up mongoose connection for MongoDB on Cloud (using X.509 pem certificate file)
    // mongoose.connect(`mongodb+srv://${encodeURIComponent('superuser_x509certificate')}@${cloud_db_server}/${db_name}`, 
    // // mongoose.connect(`mongodb+srv://${cloud_db_server}/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority`, 
    // {
    //     ssl: true,
    //     sslValidate: true,
    //     sslCA: `${__dirname}/${cloud_db_pem_file}`,
    //     authMechanism: 'MONGODB-X509', 
    //     retryWrites: true
    // }, (err) => {
    //     if(err) {
    //         console.log(`Database connection error:: ${err}`);
    //     } else {
    //         console.log('Database connected successfully on Cloud');
    //     }
    //     // TODO: exit the server in this case
    // });

    // We can also set up mongoose connection on cloud with username / password
    const cloud_db_url = `mongodb+srv://${cloud_db_username}:${cloud_db_password}@${cloud_db_server}/${db_name}`;
    mongoose.connect(cloud_db_url, {useNewUrlParser: true}, (err) => {
        if(err) {
            console.log(`Database connection error:: ${err}`);
        } else {
            console.log('Database connected successfully on Cloud');
        }
    });
} else {
    // Set up mongoose connection for MongoDB on localhost
    const local_db_url = `mongodb://${local_db_server}:${local_db_server_port}/${db_name}`;
    mongoose.connect(local_db_url, {useNewUrlParser: true}, (err) => {
        if(err) {
            console.log(`Database connection error:: ${err}`);
        } else {
            console.log('Database connected successfully on localhost');
        }
    });
}

// mongoose.connection.close();
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const userSchema = new mongoose.Schema ({
    email: String,
    password: String,
    googleId: String,
    firstname: String,
    familyname: String,
    role: String,
    department: String
});

userSchema.plugin(passportLocalMongoose);
userSchema.plugin(findOrCreate);

const User = new mongoose.model("User", userSchema);

passport.use(User.createStrategy());
passport.serializeUser((user, done) => { done(null, user.id) } );
passport.deserializeUser((id, done) => { User.findById(id, (err, user) =>  done(err, user) ) });

passport.use(new GoogleStrategy({
    clientID: process.env.GOOGLE_CLIENT_ID,
    clientSecret: process.env.GOOGLE_CLIENT_SECRET,
    callbackURL: `http://${webapp_host}:${herokuPort}/auth/google/callback`,
    userProfileURL: 'https://www.googleapis.com/oauth2/v3/userinfo'
    },
    (accessToken, refreshToken, profile, cb) => {
        console.log(profile);
        User.findOrCreate(
            {googleId: profile.id}, 
            {firstname: profile.name.givenName, familyname: profile.name.familyName, role: 'Student', department: '', email: '', password: ''}, 
            (err, user) => {
                return cb(err, user);
            }
        );
    }
));

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

app.get("/auth/google",
    passport.authenticate('google', { scope: ["profile"] })
);

app.get('/auth/google/callback', 
    passport.authenticate('google', { failureRedirect: '/login' }),
    (req, res) => {
        // authentication successful, redirect to homepage or profile page (if email is missing)
        User.findOne({googleId: req.user.googleId}, (err, foundUser) => {
            if(err) {
                console.log(`Some error occurred after google callback: ${err}`);
                res.redirect('/login');
            } else if(foundUser != null) {
                // redirect to profile page if email is missing, else redirect to homepage
                // if (foundUser.email === '' || foundUser.emailVerified === undefined || foundUser.emailVerified === '') {
                if (foundUser.email === '' ) {
                    res.redirect('/editprofile');
                } else {
                    res.redirect('/');
                }
            }
        });
    });

app.get('/logout', (req, res) => {
    req.logout(() => {
        res.redirect('/');
    });
});

app.get('/register', (req, res) => {
    if(req.isAuthenticated()) {
        res.redirect('/');
    } else {
        res.render('register');
    }
});

// app.post('/register/bcrypt', (req, res) => {
//     if(req.isAuthenticated()) {
//         res.redirect('/');
//     } else {
//     bcrypt.hash(req.body.password, saltRounds, (err, hash) => {
//         const newUser = new User({
//             email: req.body.username,
//             password: hash
//         });
//     });
//     newUser.save((err) => {
//         if (err) {
//             console.log(`Some error occurred while registering new user with Bcrypt: ${err}`);
//             res.redirect('/register/bcrypt', {errorMessage: "Some error occurred, please try again after some time!"});
//         } else {
//             res.render('login');
//         }
//     });
//     }
// });

// app.post('/login/bcrypt', (req, res) => {
//     if(req.isAuthenticated()) {
//         res.redirect('/');
//     } else {

//     const username = req.body.username;
//     const password = req.body.password;
    
//     User.findOne({email: username}, (err, foundUser) => {
//         if (err) {
//             console.log(`Some error occurred while login user with Bcrypt: ${err}`);
//             res.redirect('/login/bcrypt', {errorMessage: "Some error occurred, please try again after some time!"});
//           } else {
//             if(foundUser) {
//                 bcrypt.compare(password, foundUser.password, (err, result) => {
//                     if(result === true) {
//                         res.render('home');
//                     }
//                 });
//             }
//           }
//     });
//     }
// });

app.post('/register', (req, res) => {
    if(req.isAuthenticated()) {
        res.redirect('/');
    } else {
        User.findOne({username: req.body.username}, (err, foundUser) => {
            if(err) {
                console.log(`Some error occurred while registering: ${err}`);
                res.redirect('/register');
            } else if(foundUser != null) {
                console.log(`User ${req.body.username} already registered, redirect to Login page`);
                res.redirect('/login');
            } else {
                User.register(
                    {
                        username: req.body.username, 
                        firstname: req.body.firstname, 
                        familyname: req.body.familyname, 
                        role: req.body.role, 
                        department: req.body.department,
                        email: req.body.username
                    }, 
                    req.body.password, 
                    (err, user) => {
                        if (err) {
                            console.log(`Some error occurred while registering new user: ${err}`);
                            res.redirect('/register', {errorMessage: "Some error occurred, please try again after some time!"});
                        } else {
                            passport.authenticate('local')(req, res, () => {
                                res.redirect('/');
                        });
                        }
                });
            }
        });
    }
});

app.get('/login', (req, res) => {
    if(req.isAuthenticated()) {
        res.redirect('/');
    } else {
        res.render('login');
    }
});

app.post('/login', (req, res) => {
    if(req.isAuthenticated()) {
        res.redirect('/');
    } else {
        const user = new User({
            username: req.body.username,
            password: req.body.password
        });        
        req.login(user, (err) => {
            if (err) {
                console.log(`Some error occurred while login user with Bcrypt: ${err}`);
                res.flash('errMsg', 'Some error occurred, please try again after some time!');
                res.redirect('/login');
                // return next(err);
            } else {
                console.log(`Login ${user.username} ${user.password}`);
                // authentication and redirect to homepage
                passport.authenticate('local', { successRedirect: '/', failureRedirect: '/login', failureFlash: true, failureFlash: 'Wrong credentials entered!'})(req, res);
                // passport.authenticate('local', { successRedirect: '/', failureRedirect: '/login', failureFlash: true, failureFlash: 'Wrong credentials entered!'})(req, res, () => {
                //     console.log('Logging in... 1');
                //     res.redirect('/');
                // });
            }
        });
    }
});

app.get('/editprofile', (req, res) => {
    if(!req.isAuthenticated()) {
        res.redirect('/');
    } else {
        console.log(req.user.username);
        res.render('editprofile', {firstname: req.user.firstname, familyname: req.user.familyname, username: req.user.username, email: req.user.email, role: req.user.role, department: req.user.department});
    }
});

app.post('/editprofile', (req, res) => {
    if(!req.isAuthenticated()) {
        res.redirect('/login');
        return;
    }
    let dept = req.body.department;
    if(req.body.role === 'Student') {
        dept = '';
    }
    const updatedUser = req.user;
    updatedUser.firstname = req.body.firstname;
    updatedUser.familyname = req.body.familyname;
    updatedUser.email = req.body.email;
    updatedUser.role = req.body.role;
    updatedUser.googleId = req.user.googleId; // googleId should be blank for normal APP Login
    updatedUser.username = req.user.username; // username should be blank for Login with Google
    updatedUser.department = dept; // should be blank for Students role
    updatedUser.save( () => {
        res.redirect('/');
    });
});

app.get('/secrets', function(req, res) {
    User.find({'secret': {$ne: null}}, (err, foundUsers) => {
        if (err) {
            console.log(`Some error occurred while secrets: ${err}`);
        } else {
            if (foundUsers) {
                res.render('secrets', {usersWithSecrets: foundUsers});
            }
        }
    });
});

app.post('/submit', (req, res) => {
    const submittedSecret = req.body.secret;
    //Once user is authenticated & her session gets saved, user details are saved to req.user.
    // console.log(req.user.id);
    User.findById(req.user.id, function(err, foundUser) {
        if (err) {
            console.log(`Some error occurred while submitting: ${err}`);
        } else {
            if (foundUser) {
                foundUser.secret = submittedSecret;
                foundUser.save(function() {
                    res.redirect('/secrets');
                });
            }
        }
        });
});

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const commentSchema = {
    text: String,
    who: String,
    whoName: String,
    when: { type: Date, default: Date.now },
    status: String
};
// const Comment = mongoose.model('Comment', commentSchema);

const ticketSchema = {
    summary: {
        type: String, 
        required: [true, 'missing summary']
    }, 
    details:  {
        type: String, 
        required: [true, 'missing details']
    }, 
    owner:  {
        type: String, 
        required: [true, 'missing owner']
    }, 
    category:  {
        type: String, 
        required: [true, 'missing category']
    }, 
    department:  {
        type: String, 
        required: [true, 'missing department']
    }, 
    created:  {
        type: Date, 
        default: Date.now
    }, 
    lastUpdated: {
        type: Date,
        default: Date.now
    },
    status: {
        type: String, 
        required: [true, 'missing status']
    }, 
    isPublic: {
        type: String, 
    }, 
    tags: [ String ], 
    // comments: [{
    //     type: mongoose.ObjectId,
    //     ref: 'Comment'
    // }]
    comments: [commentSchema]
};
const Ticket = mongoose.model('Ticket', ticketSchema);

// const comment = new Schema(commentSchema);
// const Comment = mongoose.model('Comment', comment);

// const ticket = new Schema(ticketSchema);
// const Ticket = mongoose.model('Ticket', ticket);

let displaySomeMessage = '';

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

app.get('/', (req, res) => {
    if(!req.isAuthenticated()) {
        res.redirect('/login');
    } else {
        if (req.user.role === 'Student') {
            // if login user is a student, show open Tickets created by them
            Ticket.count(
                {
                    status: { $ne: 'deleted' }, 
                    owner: req.user.username
                }, (err, ticketscount) => {
                if(err) {
                    console.log(`Some error occurred while finding: ${err}`)
                    res.render('home', {ticketscount: 0, pendingcount: 0, displaySomeMessage: `Some error occurred while finding: ${err}`});
                } else {
                    console.log(`${ticketscount} tickets(s) found (Dashboard page).`)
                    res.render('home', {ticketscount: ticketscount, pendingcount: 0, resolvedcount: 0, displaySomeMessage: `${ticketscount} tickets(s) found`});
                }
            });
        } else if (req.user.role === 'Staff') {
            // if login user is staff, show open Tickets assigned to their department
            Ticket.count(
                {
                    status: { $eq: 'new' }, 
                    department: req.user.department
                }, (err, ticketscount) => {
                if(err) {
                    console.log(`Some error occurred while finding: ${err}`)
                    res.render('home', {ticketscount: 0, pendingcount: 0, displaySomeMessage: `Some error occurred while finding: ${err}`});
                } else {
                    console.log(`${ticketscount} tickets(s) found (Dashboard page).`)
                    res.render('home', {ticketscount: 0, pendingcount: ticketscount, resolvedcount: 0, displaySomeMessage: `${ticketscount} tickets(s) found`});
                }
            });
        } else {
            console.log(`Invalid user, return no data: ${req.user.username}`)
            res.render('mytickets', {tickets: [], displaySomeMessage: `Some error occurred while finding: ${req.user.username}`});
        }    
    }
});

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

app.route('/mytickets/:ticketId')
    .get((req, res) => {
        /**
         * Search ticket matching given _id. Only "one" document retrieved.
         */
        if(!req.isAuthenticated()) {
            res.redirect('/login');
        }
        Ticket.findOne(
            {
                _id: req.params.ticketId, 
                status: { $ne: 'deleted' }
            }, (err, ticket) => {
            if(err) {
                console.log(`Some error occurred: ${err}`);
                res.render('viewTicket', { tickets: []});
            } else if(ticket === null) {
                console.log(`No Ticket found matching _id ${req.params.ticketId}`);
                res.render('viewTicket', { tickets: []});
            } else {
                // console.log(`Ticket found matching _id ${req.params.ticketId}`);
                res.render('viewTicket', { tickets: [ticket]});
            }
        });
    })
    .delete((req, res) => {
        if(!req.isAuthenticated()) {
            res.redirect('/login');
        }
        res.redirect(`/mytickets/${req.params.ticketId}/delete`);
    })
    .put(async (req, res) => {
        if(!req.isAuthenticated()) {
            res.redirect('/login');
        }
        const filters = {
            _id: req.params.ticketId,
            // TODO: only owner can delete his tickets
            //_owner: 'owner_id',
            status: { $ne: 'deleted' }
        };

        // split tags string to an array of strings
        const tags = [];
        if (req.body.tags !== null) {
            req.body.tags.trim().split(',').forEach( (value) => {
                if(value.trim() !== '') {
                    tags.push(value.trim())
                }
            });
        }

        const updates = {
            $set: {
            summary: req.body.summary,
            details: req.body.details,
            category: req.body.category,
            department: req.body.department,
            lastUpdated: moment(new Date()).format('llll'),
            status: 'modified',
            isPublic: 'No',
            tags: tags
        }};
        let newDoc = await Ticket.findOneAndUpdate(filters, updates, {returnOriginal: false});
        // console.log(newDoc);
        res.redirect('/mytickets/:ticketId');
    });

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

app.route('/mytickets')
    .get((req, res) => {
        /**
         * Search all tickets (except those marked as deleted).
         */
        if(!req.isAuthenticated()) {
            res.redirect('/login');
            return;
        }
        const createNewTicket = req.query.createNew;    
        if(createNewTicket) {
            // Render createticket.ejs page
            res.render('createticket');
        } else {
            if (req.user.role === 'Student') {
                // if login user is a student, show open Tickets created by them
                Ticket.find(
                    {
                        status: { $ne: 'deleted' }, 
                        owner: req.user.username
                    }, (err, tickets) => {
                    if(err) {
                        console.log(`Some error occurred while finding: ${err}`)
                        res.render('mytickets', {tickets: [], displaySomeMessage: `Some error occurred while finding: ${err}`});
                    } else {
                        console.log(`${tickets.length} tickets(s) found (Tickets page).`)
                        res.render('mytickets', {tickets: tickets, displaySomeMessage: `${tickets.length} ticket(s) found`});
                    }
                }).sort({status: 'asc', created: 'desc'});
            } else if (req.user.role === 'Staff') {
                // if login user is staff, show open Tickets assigned to their department
                Ticket.find(
                    {
                        status: { $ne: 'deleted' }, 
                        department: req.user.department
                    }, (err, tickets) => {
                    if(err) {
                        console.log(`Some error occurred while finding: ${err}`)
                        res.render('mytickets', {tickets: [], displaySomeMessage: `Some error occurred while finding: ${err}`});
                    } else {
                        console.log(`${tickets.length} tickets(s) found (Tickets page).`)
                        res.render('mytickets', {tickets: tickets, displaySomeMessage: `${tickets.length} ticket(s) found`});
                    }
                }).sort({status: 'asc', created: 'desc'});
            } else {
                console.log(`Invalid user, return no data: ${req.user.username}`)
                res.render('mytickets', {tickets: [], displaySomeMessage: `Error: user.role is missing`});
            }
        }
    })
    .post((req, res) => {
        /** 
         * Receive post request (from createticket.ejs page), and create the ticket
        */
        if(!req.isAuthenticated()) {
            res.redirect('/login');
        }

        // split tags string to an array of strings
        const tags = [];
        if (req.body.tags !== null) {
            req.body.tags.trim().split(',').forEach( (value) => {
                if(value.trim() !== '') {
                    tags.push(value.trim())
                }
            });
        }
        
        const timestamp = moment(new Date()).format('llll');
        const ticket = new Ticket({
            summary: req.body.summary, 
            details: req.body.details, 
            owner: req.user.username, 
            category: req.body.category, 
            department: req.body.department, 
            created: timestamp,
            lastUpdated: timestamp,
            status: 'new',
            isPublic: 'No', 
            tags: tags
        });
    
        ticket.save((err, newId) => {
                if(err) {
                    console.log('Some error occurred while creating new ticket: ' + err);
                    res.send(err);
                } else {
                    console.log('New ticket created with ID ' + newId._id);
                    res.redirect('/mytickets/' + newId._id);
                }
            }
        );
    });

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
* Receive post request (from createticket.ejs page), and change status of Ticket to deleted
*/
app.post('/mytickets/:ticketId/delete', (req, res) => {
    if(!req.isAuthenticated()) {
        res.redirect('/login');
    }
    Ticket.findOne(
        {
            _id: req.params.ticketId, 
            // TODO: only owner can delete his tickets
            //_owner: 'owner_id',
            status: { $ne: 'deleted' }
        }, async (err, tickets) => {
        if(err) {
            console.log(`Some error occurred: ${err}`);
            res.render('viewTicket', { tickets: []});
        } else if(tickets === null) {
            console.log(`No Ticket found matching _id ${req.params.ticketId}`);
            res.render('viewTicket', { tickets: []});
        } else {
            console.log(`Deleting ticket with _id ${req.params.ticketId}`);
            const filter = { _id: req.params.ticketId };
            const update = { status: 'deleted' };
            let newDoc = await Ticket.findOneAndUpdate(filter, update, {returnOriginal: false});
            // console.log(newDoc);
            res.redirect('/mytickets');
        }
    });
});

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

app.post('/mytickets/:ticketId/addcomment', async (req, res) => {
    if(!req.isAuthenticated()) {
        res.redirect('/login');
    } else {
        console.log(`Adding comment [${req.params.comment}] to ticket _id ${req.params.ticketId}`);
        if(req.params.comment !== null && req.params.comment !== '') {
            const comment = {
                text: req.body.addcomment,
                who: req.user.username,
                whoName: `${req.user.firstname} ${req.user.familyname}`,
                when: moment(new Date()).format('llll'),
                status: 'new'
            };
            const filter = { _id: req.params.ticketId };
            // let newDoc2 = await Ticket.updateOne(filter, {$addToSet: {comments: comment}}, {returnOriginal: false});
            // let newDoc = await Ticket.findOneAndUpdate(filter, update, {returnOriginal: false});
    
            const myTicket = await Ticket.findOne(filter);
    
            if(myTicket !== null && myTicket.owner === req.user.username) {
                myTicket.comments.push(comment);
                await myTicket.save();
                console.log("Ticket: " + myTicket);
            } else {
                console.log(`Failed to add comments because user [${req.user.username}] does not own ticket _id: [${req.params.ticketId}]`);
            }
        }
        res.redirect(`/mytickets/${req.params.ticketId}`);
    }
});

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * My Enropllements
 */
app.get('/myenrolls', (req, res) => {
    if(!req.isAuthenticated()) {
        res.redirect('/login');
    } else {
        res.render('myenrolls', {myenrolls: null, displaySomeMessage: `You have no enrollments!`});
    }
});

app.get('/about', (req, res) => {
    res.render('about', {
        contactContent: 'Contact Ajay on amalik007@gmail.com or +1 (306) 361 4491'
    });
});

/**
 * Spin NodeJS web server on port 3000
 */
app.listen(herokuPort, function() {
	// host and port are defined in ./modules/config.json and loaded in this file via require module
	console.log(`Node server started at http://${webapp_host}:${herokuPort}`);
});
