const gulp = require('gulp');
const sass = require('gulp-sass')(require('sass'));
const uglify = require('gulp-uglify');
const rename = require('gulp-rename');

// Compile Sass
gulp.task('sass', function() {
    return gulp.src('src/scss/*.scss')
        .pipe(sass().on('error', sass.logError))
        .pipe(gulp.dest('dist/css'));
});

// Minify JavaScript
gulp.task('scripts', function() {
    return gulp.src('src/js/*.js')
        .pipe(uglify())
        .pipe(rename({ suffix: '.min' }))
        .pipe(gulp.dest('dist/js'));
});

// Watch files for changes
gulp.task('watch', function() {
    gulp.watch('src/scss/*.scss', gulp.series('sass'));
    gulp.watch('src/js/*.js', gulp.series('scripts'));
});

// Default task
gulp.task('default', gulp.series('sass', 'scripts', 'watch'));
