const mongoose = require('mongoose')
const RawTweet = new mongoose.Schema({
    data: Object,
    scrape_date: {
        required: true,
        default: Date.now(),
        type: Date
    }
})


const model = mongoose.model('RawTweet', RawTweet)

module.exports = model