const setup = require('./db');
const scrape = require('./scraper');
const RawTweet = require('./models/RawTweet');
const chalk = require('chalk');

const log = console.log;

setup()
  .then(async (conn) => {
    const parameters = {
      from: '2020-04-01',
      to: '2021-08-01',
      query: `"online class" OR "e-class" OR "online classes"`,
      location: 'Cebu',
      lang: 'en',
    };

    const { filtered } = await scrape(parameters);

    const len = RawTweet.find().length



    for (const tweet of filtered) {
      try {
        const { id } = tweet;
        const dup = RawTweet.findOne({ tweet_id: id });

        if (!dup || !len) {
          const toSave = new RawTweet({
            data: tweet,
            parameters,
            tweet_id: id,
          });
          toSave.save();
        }
      } catch (error) {
        continue;
      }
    }

    log(chalk.green('\nDATA') + ' saved');
  })
  .catch((error) => {
    console.log(error);
    process.exit();
  });
