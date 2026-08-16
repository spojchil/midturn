//! 模型替身：一次性 / 流式受控模型，静态提示源，流观察端。

use super::super::*;

pub(crate) struct StaticPrompt;

impl PromptSource for StaticPrompt {
    fn base_context(&self) -> Result<Vec<TranscriptItem>, AgentError> {
        Ok(vec![
            input("system", "base"),
            input("developer", "base-constraint"),
        ])
    }
}

pub(crate) struct ChannelModel {
    pub(crate) requests: mpsc::UnboundedSender<ModelRequest>,
    pub(crate) responses: Mutex<mpsc::UnboundedReceiver<Result<ModelResponse, AgentError>>>,
}

impl Model for ChannelModel {
    fn complete<'a>(
        &'a self,
        request: ModelRequest,
    ) -> PortFuture<'a, Result<ModelResponse, AgentError>> {
        Box::pin(async move {
            self.requests.send(request).unwrap();
            self.responses.lock().await.recv().await.unwrap()
        })
    }
}

pub(crate) enum StreamCommand {
    Event(ModelStreamEvent),
    Complete(ModelResponse),
    Fail(AgentError),
}

pub(crate) struct CommandStreamModel {
    pub(crate) requests: mpsc::UnboundedSender<ModelRequest>,
    pub(crate) commands: Mutex<mpsc::UnboundedReceiver<StreamCommand>>,
}

impl Model for CommandStreamModel {
    fn complete<'a>(
        &'a self,
        _request: ModelRequest,
    ) -> PortFuture<'a, Result<ModelResponse, AgentError>> {
        Box::pin(async { panic!("测试流模型不应走 one-shot complete") })
    }

    fn complete_stream<'a>(
        &'a self,
        request: ModelRequest,
        sink: &'a mut dyn ModelStreamSink,
    ) -> PortFuture<'a, Result<ModelResponse, AgentError>> {
        Box::pin(async move {
            self.requests.send(request).unwrap();
            loop {
                let command = {
                    let mut commands = self.commands.lock().await;
                    commands.recv().await.unwrap()
                };
                match command {
                    StreamCommand::Event(event) => sink.emit(event).await?,
                    StreamCommand::Complete(response) => return Ok(response),
                    StreamCommand::Fail(error) => return Err(error),
                }
            }
        })
    }
}

#[derive(Default)]
pub(crate) struct RecordingStreamObserver {
    pub(crate) events: StdMutex<Vec<ObservedModelStreamEvent>>,
}

impl StreamObserver for RecordingStreamObserver {
    fn observe(&self, event: &ObservedModelStreamEvent) {
        self.events.lock().unwrap().push(event.clone());
    }
}
