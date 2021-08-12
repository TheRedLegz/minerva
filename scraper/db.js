const mongoose = require('mongoose');
const chalk = require('chalk')

// TODO look into cleaning the tweet here in js or python

const setup = () =>
  new Promise(async (resolve, reject) => {
    try {
      mongoose.connection.on('connected', () => console.log(chalk.green('DATABASE') + ' connected\n'));
      mongoose.connection.on(
        'disconnected',
        () => console.log(chalk.red('DATABASE') + ' disconnected')
      );

      await mongoose.connect('mongodb://localhost:27017/minerva_raw_tweets', {
        useNewUrlParser: true,
        useUnifiedTopology: true,
      });

      const connection = mongoose.connection;

      resolve(connection);
      
    } catch (error) {
      reject(error);
    }
  });

module.exports = setup;
