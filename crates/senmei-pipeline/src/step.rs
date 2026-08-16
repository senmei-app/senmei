use senmei_media::Frame;

pub trait Step: Send {
    fn name(&self) -> &'static str;
    fn process(&mut self, frame: &mut Frame) -> crate::Result<()>;
}

pub struct Passthrough;

impl Step for Passthrough {
    fn name(&self) -> &'static str {
        "passthrough"
    }

    fn process(&mut self, _frame: &mut Frame) -> crate::Result<()> {
        Ok(())
    }
}
