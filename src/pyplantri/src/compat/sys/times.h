/* Win32 subset for plantri_sqs; child-process CPU accounting is not supported.
 * Interface reference (declarations only; independent implementation below):
 * https://github.com/coreutils/gnulib/blob/master/lib/sys_times.in.h
 */
#pragma once
#include <time.h>
#include <windows.h>
#ifndef CLK_TCK
#define CLK_TCK CLOCKS_PER_SEC
#endif

struct tms { clock_t tms_utime, tms_stime, tms_cutime, tms_cstime; };

/* FILETIME counts 100 ns intervals; all results use CLK_TCK ticks per second. */
static clock_t sqs_filetime_ticks(FILETIME value) {
    ULONGLONG ticks = ((ULONGLONG)value.dwHighDateTime << 32) | value.dwLowDateTime;
    return (clock_t)(ticks * CLK_TCK / 10000000);
}

static clock_t times(struct tms *buffer) {
    FILETIME created, exited, kernel, user;
    buffer->tms_utime = buffer->tms_stime = buffer->tms_cutime = buffer->tms_cstime = 0;
    if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user))
        return (clock_t)-1;
    buffer->tms_utime = sqs_filetime_ticks(user);
    buffer->tms_stime = sqs_filetime_ticks(kernel);
    return (clock_t)(GetTickCount64() * CLK_TCK / 1000);
}
