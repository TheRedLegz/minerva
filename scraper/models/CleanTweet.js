const mongoose = require('mongoose')


const schema = new mongoose.Schema({
    text: {
        required: true,
        type: String
    },
    geo: {
        required: false,
        type: String
    },
    user: {
        required: true,
        type: String
    },
    tweet_id: {
        required: true,
        type: String
    },
    tweet_date: {
        required: true,
        type: Date
    },
})


const model = mongoose.model('CleanTweet', schema)



module.exports = model