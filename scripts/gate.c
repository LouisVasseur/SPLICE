// SPDX-License-Identifier: MIT
// LD_PRELOAD for GRE's microbench: parks the process in a blocking open() of a FIFO, so a built index waits with no CPU
// and no memory traffic until gated_run.py releases it. The gate is the first of: getenv("TRANSFORM_WARM"), which w_<name>
// calls right before its untimed warm-up (include/gret/warmup.h), so the warm-up runs after the release; and "Begin
// running", the last output before GRE's clock starts. The binary is unchanged and the timed loop calls neither. Without
// $GATE this does nothing.
//   gcc -shared -fPIC -O2 -Wall gate.c -o gate.so -ldl
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

extern char **environ;

static void park(void) {
    static int parked;
    const char *g = getenv("GATE");
    if (parked || !g) return;
    parked = 1;
    char go[PATH_MAX], ready[PATH_MAX], c;
    snprintf(go, sizeof go, "%s.go", g);
    snprintf(ready, sizeof ready, "%s.ready", g);
    int fd = mkfifo(go, 0600) == 0 ? open(ready, O_CREAT | O_WRONLY, 0600) : -1;
    if (fd < 0) { perror("gate"); _exit(3); }
    close(fd);
    fd = open(go, O_RDONLY);
    if (fd < 0 || read(fd, &c, 1) != 1) { perror("gate"); _exit(3); }
    close(fd);
}

char *getenv(const char *name) {  // no dlsym here: jemalloc calls getenv while it starts, and dlsym may allocate
    if (strcmp(name, "TRANSFORM_WARM") == 0) park();
    size_t n = strlen(name);
    for (char **e = environ; e && *e; e++)
        if (strncmp(*e, name, n) == 0 && (*e)[n] == '=') return *e + n + 1;
    return NULL;
}

int puts(const char *s) {
    static int (*real)(const char *);
    if (!real) real = (int (*)(const char *))dlsym(RTLD_NEXT, "puts");
    if (strcmp(s, "Begin running") == 0) park();
    return real(s);
}
