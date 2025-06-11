---
layout: post
title: Modern C++ features
description: Brief notes of modern C++ features
tags: C++ Boost coroutine
giscus_comments: true
date: 2025-6-11
featured: false
---
# Coroutine
Coroutine can be seem as "functions whose execution you can pause" and later "resumed". C++ coroutines are stackless: they suspend execution by returning to the caller, and the data that is required to resume execution is stored separately from the stack. 

![Co-Routine](/assets/img/Co-routine.png){:width="70%"}

C++ 20 supported with following concepts:

* **co_await**: to suspend execution until resumed, the arguments for co_await should be **awaitable** instance, which will be explained later.
{% highlight c++ linenos %}
task<> tcp_echo_server()
{
    char data[1024];
    while (true)
    {
        std::size_t n = co_await  socket.async_read_some(buffer(data));  //suspend the execution and return to the caller
                        co_await  async_write(socket, buffer(data, n));  //suspend the execution and return to the caller
    }
}

{% endhighlight %}


* **co_yeild**: to suspend execution while returning a value:
{% highlight c++ linenos %}
generator<unsigned int> iota(unsigned int n = 0)
{
    while (true)
        co_yield n++;
}

{% endhighlight %}

* **co_yeild**: complete execution and returning a value
{% highlight c++ linenos %}
lazy<int> f()
{
    co_return 7;
}
{% endhighlight %}

## promise in coroutine
Any function is a coroutine if it contains any of 'co_wait', 'co_yield' or 'co_return'. For coroutine contains 'co_wait', the return type should alias the special **promise_type** (via **using** or typedef like in general) to the specific class or struct defined by the user. The aliased specific class is generally called **promise**, which must implement some of the specific functions used by the caller to control execution of the coroutine . Coroutine submits its result or exception through this promise object. 

{% highlight c++ linenos %}

class coroutine_return_obj

struct my_promise
{
    //should return object that the coroutine returns
    coroutine_return_obj get_return_object() noexcept {
        return coroutine_return_obj { std::coroutine_handle<my_promise>::from_promise(*this) };
    };

    //Called before the co_wait() call.
    std::suspend_never initial_suspend()  const noexcept { return {}; }

    //Called just before the coroutine ends.
    std::suspend_never final_suspend()    const noexcept { return {}; }

    //Called when no co_returned is called at the end.
    void return_void() noexcept {}

     //Called when some exception happens in coroutine.
    void unhandled_exception() noexcept {
        std::cerr << "Unhandled exception caught...\n";
        exit(1);
    }
};
{% endhighlight %}