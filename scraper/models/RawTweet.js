const mongoose = require('mongoose')
const RawTweet = new mongoose.Schema({
    data: Object
})


const model = mongoose.model('RawTweet', RawTweet)

module.exports = model