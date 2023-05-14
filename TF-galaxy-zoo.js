#Galaxy zoo descriptor for all the marbles
print 1,2,3,4,
name: (galaxy-barndogger) 
version: (0.0.1) 
entry point: (index.js) 
keywords: 
license: (ISC) 
About to write to /Users/jefferyschmitz/tensorflow/tensorflow/models/image/imagenet/Galaxy-Zoo/package.json:

{
  "name": "galaxy-zoo",
  "private": true,
  "version": "0.0.1",
  "author": "zooniverse",
  "engineStrict": true,
  "engine-strict": true,
  "engines": {
    "node": "<= 0.11.14"
  },
  "repository": {
    "type": "git",
    "url": "git+https://github.com/zooniverse/Galaxy-Zoo.git"
  },
  "scripts": {
    "start": "hem server --port 6414",
    "pretest": "hem server --port 9290 & echo $! > ./hem.pid",
    "test": "jasmine-phantom-node --port 9290",
    "posttest": "kill `cat hem.pid`; rm hem.pid",
    "seed-locale": "seed-translation --project galaxy_zoo --env production --deploy-path locales"
  },
  "dependencies": {
    "hem": "git://github.com/edpaget/hem.git#83617d1a4e",
    "node-pubsub": "*",
    "nib": "0.9.1",
    "es5-shimify": "~0.0.1",
    "json2ify": "~0.0.1",
    "jqueryify": "~0.0.1",
    "underscore": "*",
    "spine": "~1.0.7",
    "clean-css": "0.6.0",
    "uglify-js": "1.3.4",
    "ubret": "git://github.com/zooniverse/Ubret.git#galaxy_zoo",
    "zooniverse": "git://github.com/zooniverse/Zooniverse.git#galaxy_zoo",
    "coffee-script": "1.6.2",
    "translator-seed": "0.1.1"
  },
  "devDependencies": {
    "jasmine-phantom-node": "git://github.com/edpaget/jasmine-phantom-node.git"
  },
  "description": "### Getting Started",
  "bugs": {
    "url": "https://github.com/zooniverse/Galaxy-Zoo/issues"
  },
  "homepage": "https://github.com/zooniverse/Galaxy-Zoo#readme",
  "main": "index.js",
  "directories": {
    "test": "test"
  },
  "license": "ISC"
}
