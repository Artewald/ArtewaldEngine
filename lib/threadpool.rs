use std::{thread::available_parallelism, sync::{Arc, RwLock}};

#[derive(Debug, Clone, Copy)]
pub enum ThreadState {
    Free,
    Used,
    Done,
}

#[derive(Debug, Clone)]
pub struct ThreadPoolHelper {
    max_num_threads: usize,

    num_utilized: Arc<RwLock<usize>>,
}

#[allow(dead_code)]
impl ThreadPoolHelper {
    pub fn new(max_num_threads: Option<usize>) -> Arc<RwLock<ThreadPoolHelper>> {
        let threads = max_num_threads.unwrap_or(available_parallelism().unwrap().get());
        if  threads > available_parallelism().unwrap().get() {
            panic!("ThreadPoolHelper::new() -> You cannot have more threads than you have: you specified {}, but only {} are available.", threads, available_parallelism().unwrap().get());
        }
        Arc::new(RwLock::new(ThreadPoolHelper { max_num_threads: threads, num_utilized: Arc::new(RwLock::new(0))}))
    }

    /// Returns true if a thread was started, false if not
    pub fn try_starting_thread(&self) -> bool {
        if self.max_num_threads <= *self.num_utilized.clone().read().unwrap() {
            return false;
        }
        *self.num_utilized.clone().write().unwrap() += 1;
        // if self.max_num_threads <= *self.num_utilized.clone().read().unwrap() {
        //     *self.num_utilized.clone().write().unwrap() -= 1;
        //     return false;
        // }
        true
    }

    pub fn end_thread(&self) {
        if *self.num_utilized.clone().read().unwrap() == 0 {
            panic!("ThreadPoolHelper::end_thread() -> Something went horribly wrong. There are no threads to end!")
        }
        *self.num_utilized.clone().write().unwrap() -= 1;
    }

    pub fn available_threads(&self) -> usize {
        if self.max_num_threads < *self.num_utilized.clone().read().unwrap() {
            panic!("There are more utilized/started threads then the max amount");
        }
        self.max_num_threads - *self.num_utilized.clone().read().unwrap()
    }
    
    pub fn print_num_utilized(&self) {
        println!("Num threads utilized: {}", *self.num_utilized.clone().read().unwrap());
    }
}