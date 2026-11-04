"""Engine modules whose top-level code runs an installer that sets a watched process global.

Both call kernel_v30_token_contracts.install_telemetry_patch(), which sets
kernel_measurement._v30_accounting_patch. conftest imports them before snapshotting the pristine installer
baseline; tests/test_import_time_installers.py fails if this list falls out of date.
"""

IMPORT_TIME_INSTALLER_MODULES = ("kernel_model_policy", "kernel_stateframe_runtime")
