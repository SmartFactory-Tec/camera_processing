const RESTART_TIME_HOURS = 24;
const RESTART_TIME_MS = RESTART_TIME_HOURS * 60 * 60 * 1000;

const kill  = require('tree-kill');
const cproc = require("child_process");

const ERROR = 'ERROR';

let proc; // child.ChildProcess;
let timeoutRef;

function start() {
    console.log(__dirname + "/main.py")
    proc = cproc.spawn("python3", [__dirname + "/main.py"]);
    proc.stdout?.on('data', function(res) {
        if (res.toString().indexOf(ERROR) !== -1) {
            if (timeoutRef) {
                clearTimeout(timeoutRef)
            }
            restart()
        }
        console.log(res.toString());
    });
    proc.stderr?.on('data', function(res) {
        if (res.toString().indexOf(ERROR) !== -1) {
            if (timeoutRef) {
                clearTimeout(timeoutRef)
            }
            restart()
        }
        console.log(res.toString());
    });
    proc.stderr?.on('exit', function(code) {
        if (timeoutRef) {
            clearTimeout(timeoutRef)
        }
        restart()
    });
    timeoutRef = setTimeout(restart, RESTART_TIME_MS);
}

function restart() {
    kill(proc.pid);
    start();
}

start();
