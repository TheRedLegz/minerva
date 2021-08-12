const mongoose = require('mongoose');
const RawTweet = new mongoose.Schema({
  data: Object,
  scrape_date: {
    required: true,
    default: Date.now(),
    type: Date,
  },
  parameters: {
    required: true,
    type: Object
  },
  tweet_id: {
    required: true,
    type: String,
    index: true,
    unique: true
  }
});

const model = mongoose.model('RawTweet', RawTweet);

module.exports = model;
