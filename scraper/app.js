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
      query: `"online class" OR "online classes" OR "internet"`,
      location: 'Cebu',
      lang: 'en',
    };

    const { filtered } = await scrape(parameters);

    for (const tweet of filtered) {
      const { id } = tweet;
      const dup = await RawTweet.findOne({ tweet_id: id });
      
      if (!dup) {
        const toSave = new RawTweet({
          data: tweet,
          parameters,
          tweet_id: id,
        });
        toSave.save();
      }
    }

    log(chalk.green('\nDATA') + ' saved');
  })
  .catch((error) => {
    console.log(error);
    process.exit();
  });
