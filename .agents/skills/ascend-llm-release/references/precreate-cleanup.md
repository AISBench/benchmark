# Pre-create failure cleanup

Local and SSH node cleanup no longer release claims based only on an empty
coordinator container map. HostGuard records effect_kind with durable command
acceptance. release_unstarted checks the owner's host-wide command history while
holding the resource lock. It releases only matching claims and only when no
create/unknown effect has been accepted.

A legacy record lacking classification is unknown, as is a failed command whose
external effect was not confirmed. Both retain claims and require resource
inventory verification. This prevents an interrupted create from being hidden by
a later coordinator failure before its local create phase. Successful pre-create
cleanup records its basis as no durable create acceptance.

The remote worker derives resource identities from frozen groups and uses the
same host guard; the control side cannot supply an arbitrary set to unlock.
Tests exercise persistent claims, unknown create outcomes and conflicting tasks
locally. This does not establish actual Docker/SSH crash recovery on a server.
