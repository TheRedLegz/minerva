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

    const contents = await RawTweet.find({}).exec();
    const length = contents.length;

    for (const tweet of filtered) {
      const { id } = tweet;
      const dup = RawTweet.findOne({ tweet_id: id });

      if (!dup || !length) {
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
