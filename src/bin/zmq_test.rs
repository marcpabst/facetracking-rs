use nng::{Aio, AioResult, Context, Protocol, Socket};


const ADDRESS: &'static str = "inproc://nng/example";


fn server() -> Result<(), nng::Error> {
    // Set up the server and listen for connections on the specified address.
    let s = Socket::new(Protocol::Rep0)?;

    // Set up the callback that will be called when the server receives a message
    // from the client.
    let ctx = Context::new(&s)?;
    let ctx_clone = ctx.clone();
    let aio = Aio::new(move |aio, res| worker_callback(aio, &ctx_clone, res))?;

    s.listen(ADDRESS)?;

    // start the worker thread
    ctx.recv(&aio)?;

    println!("Server listening on {}", ADDRESS);

    loop {
        println!(".");
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

}

fn client() -> Result<(), nng::Error> {
    // Set up the client and dial the specified address.
    let s = Socket::new(Protocol::Req0)?;
    s.dial(ADDRESS)?;

    // Send a message to the server.
    loop {

        let timestamp_micros = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros();

        let msg = timestamp_micros.to_be_bytes();
        s.send(msg)?;
   
        println!("Client sent message at {}", timestamp_micros);

        // Wait for 1 second before sending the next message.
        std::thread::sleep(std::time::Duration::from_secs(1));
    }

}

fn worker_callback(aio: Aio, ctx: &Context, res: AioResult) {
    // This is the callback that will be called when the worker receives a message
    // from the client. This fnction just prints the message.
    match res {
         // We successfully received a message.
         AioResult::Recv(m) => {
            let msg = m.unwrap();

            // extract the timestamp from the message (use from_be_bytes to convert from big endian)
            let timestamp_micros = u128::from_be_bytes(msg.as_slice().try_into().unwrap());
            
            let timestamp_micros_now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros();
            println!("Worker received a message. Delay: {} micros", timestamp_micros_now - timestamp_micros);
            ctx.recv(&aio).unwrap();
        }
        // We successfully sent a message.
        AioResult::Send(m) => {
            println!("Worker sent message");
        }
        // We are sleeping.
        AioResult::Sleep(r) => {
            println!("Worker sleeping");
        } 

    }
}

fn main() -> Result<(), nng::Error> {
    let _ = std::fs::remove_file(ADDRESS);
    let server = std::thread::spawn(|| server());
    let client = std::thread::spawn(|| client());
  
    client.join().unwrap()?;
    server.join().unwrap()?;

    // wait 10 seconds
    std::thread::sleep(std::time::Duration::from_secs(10));

    Ok(())
}
