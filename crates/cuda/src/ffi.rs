use std::ffi::CStr;

use anyhow::{Result, anyhow};

pub fn ffi_wrap<F>(mut inner: F) -> Result<()>
where
    F: FnMut() -> *const std::os::raw::c_char,
{
    unsafe extern "C" {
        fn free(str: *const std::os::raw::c_char);
    }

    let c_ptr = inner();
    if c_ptr.is_null() {
        Ok(())
    } else {
        let what = unsafe {
            let msg = CStr::from_ptr(c_ptr)
                .to_str()
                .unwrap_or("Invalid error msg pointer")
                .to_string();
            free(c_ptr);
            msg
        };
        Err(anyhow!(what))
    }
}
